import sys, json
from scipy.optimize import minimize
import numpy as np
import jax
#jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import jax_md
import matplotlib.pyplot as plt
import jax_md.amber.amber_energy as amber
#import parmed as pmd
import openmm as omm
import openmm.app as app
jax.config.update("jax_enable_x64", True)
import argparse
from jaxreaxff.smartformatter import SmartFormatter

from parmedmod import UpdateParmTopInMemory
import os, re, glob, shutil

def ConvertMultiPrimParamsForParmEd(params_dict):
    """
    Convert multi-primitive parameter format for ParmEd.

    Input format (from optimizer):
        {'9-10-11-12_p1': {'height': 1.0, 'phase': 0.0, 'periodicity': 1, ...},
         '9-10-11-12_p2': {'height': 0.5, 'phase': 0.0, 'periodicity': 2, ...},
         '9-10-11-12_p3': {'height': 0.3, 'phase': 0.0, 'periodicity': 3, ...}}

    Output format (for ParmEd):
        {'9-10-11-12': {'height': [1.0, 0.5, 0.3],
                        'phase': [0.0, 0.0, 0.0],
                        'periodicity': [1, 2, 3],
                        'scee': 1.2, 'scnb': 2.0,
                        'torsion_mask': [...]}}
    """
    grouped = {}

    for key, value in params_dict.items():
        # Extract base key (remove _pN suffix if present)
        if '_p' in key:
            base_key = key.split('_p')[0]
            periodicity = int(key.split('_p')[1])
        else:
            # Single primitive (no suffix)
            base_key = key
            periodicity = value.get('periodicity', 1)

        # Initialize base key if not seen
        if base_key not in grouped:
            grouped[base_key] = {
                'height': [],
                'phase': [],
                'periodicity': [],
                'scee': value.get('scee', 1.2),  # Scalar (same for all primitives)
                'scnb': value.get('scnb', 2.0),  # Scalar (same for all primitives)
                'torsion_mask': value.get('torsion_mask', [])  # Same for all primitives
            }

        # Append to lists (order by periodicity to ensure [1,2,3] not [2,1,3])
        grouped[base_key]['height'].append(value['height'])
        grouped[base_key]['phase'].append(value['phase'])
        grouped[base_key]['periodicity'].append(value.get('periodicity', periodicity))

    # Sort by periodicity for each torsion
    for base_key in grouped:
        # Zip together, sort by periodicity, unzip
        combined = list(zip(
            grouped[base_key]['periodicity'],
            grouped[base_key]['height'],
            grouped[base_key]['phase']
        ))
        combined.sort(key=lambda x: x[0])  # Sort by periodicity

        periodicities, heights, phases = zip(*combined) if combined else ([], [], [])
        grouped[base_key]['periodicity'] = list(periodicities)
        grouped[base_key]['height'] = list(heights)
        grouped[base_key]['phase'] = list(phases)

    return grouped

# make global array for loss
losses = []
iteration = 0

best_loss = jnp.iinfo(jnp.int64).max
best_params = None
best_iteration = -1
best_energy = None # AB: Track the energy for the best iteration.

# Reads json data
def ReadJsonData(json_path):

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    return json_data

# AB:  here you control how many windows you sample over 360°
params = ReadJsonData('params.json')
NPOINTS = params['npoints']

# dumps json data into an existing file
def SaveJsonData(field_dict, json_path):

    with open(json_path, 'r') as f:
        json_data = json.load(f)

    for key in field_dict:
        json_data[key]=field_dict[key]

    with open(json_path, 'w') as outfile:
        json.dump(json_data, outfile)

def extractCoordinates(flist):
    coordinates = []

    for file in flist:
        with open(file, 'r') as f:
            lines=f.readlines()

        crds = []
        for line in lines[2:]:
            crds.append([jnp.float32(i) for i in line.split()[1:]])
        #print(crds)
        crds = jnp.array(crds)
        #A -> NM
        coordinates.append(crds/10)

    return coordinates

# AB: Clean the outdir folder and save best iteration to the orriginal geo_dir path. 
def cleanup_and_restore_best(outdir, dest_dir, best_iteration):
    """
    Copies the contents of the best iteration folder (iteration_<best_iteration>) from outdir 
    back to dest_dir, then deletes all iteration folders in outdir.
    
    Parameters:
      outdir: the directory where iteration folders are stored.
      dest_dir: the destination directory (os.path.dirname(geo_dir)) where the best iteration files should be copied.
      best_iteration: the iteration number corresponding to the best iteration.
    """
    # Define pattern for iteration folders in outdir.
    pattern = os.path.join(outdir, "iteration_*")
    iteration_folders = glob.glob(pattern)
    
    best_folder = os.path.join(outdir, f"iteration_{best_iteration}")
    if not os.path.exists(best_folder):
        print("Best iteration folder not found in outdir!")
        return

    # Copy all files from best_folder back to dest_dir.
    for root, dirs, files in os.walk(best_folder):
        for file in files:
            src = os.path.join(root, file)
            # Compute the relative path from best_folder to this file.
            rel_path = os.path.relpath(src, best_folder)
            dst = os.path.join(dest_dir, rel_path)
            # Create destination directories if needed.
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

    # Delete all iteration folders (only directories) in outdir.
    for folder in iteration_folders:
        if os.path.isdir(folder):
            shutil.rmtree(folder)


def constrained_minimization_vec(crds, prmtop, boxVectors, min_steps, torsions, min_interval):
    radian_to_degree = 180.0/jnp.pi
    degree_to_radian = 1.0/radian_to_degree
    bondprm = amber.bond_init(prmtop._prmtop)
    angleprm = amber.angle_init(prmtop._prmtop)
    torsionprm = amber.torsion_init(prmtop._prmtop)
    ljprm = amber.lj_init(prmtop._prmtop)
    coulprm = amber.coul_init(prmtop._prmtop)
    prms = (bondprm, angleprm, torsionprm, ljprm, coulprm)
    # restraint format:
    # p1,p2,p3,p4,resangle(radians),frc1,frc2
    # default frc1/frc2 values - 1000.0/0.25
    # frc1 = 1000
    # frc2 = .25
    # t = [6,7,9,11]

    def energy_fn(pos, prms=None, restraint=None):
        bprm, aprm, tprm, lprm, cprm, = prms
        return jnp.float32((amber.bond_get_energy(pos, boxVectors, bprm) \
                + amber.angle_get_energy(pos, boxVectors, aprm) \
                + amber.torsion_get_energy(pos, boxVectors, tprm) \
                + amber.lj_get_energy(pos, boxVectors, lprm) \
                + amber.coul_get_energy(pos, boxVectors, cprm) \
                + amber.rest_get_energy(pos, boxVectors, restraint=restraint))/4.184)
                #+ 0)/4.184

    def energy_fn_no_restraint(pos, prms=None, restraint=None):
        bprm, aprm, tprm, lprm, cprm, = prms
        return jnp.float32((amber.bond_get_energy(pos, boxVectors, bprm) \
                + amber.angle_get_energy(pos, boxVectors, aprm) \
                + amber.torsion_get_energy(pos, boxVectors, tprm) \
                + amber.lj_get_energy(pos, boxVectors, lprm) \
                + amber.coul_get_energy(pos, boxVectors, cprm) \
                #+ amber.rest_get_energy(pos, boxVectors, restraint=restraint))/4.184)
                + 0)/4.184)

    masses = jnp.array([jnp.float32(val) for val in prmtop._prmtop._raw_data['MASS']])
    displacement_fn, shift_fn = jax_md.space.periodic_general(boxVectors, fractional_coordinates=False)
    key = jax.random.PRNGKey(0)
    energy_fn = jax.jit(energy_fn)
    init_fn, apply_fn = jax_md.minimize.fire_descent(energy_fn, shift_fn, 1e-3, 1e-3)
    #state = init_fn(positions, mass=masses)

    def body_fn(i, stateList):
        state, restraint, prms = stateList
        state = apply_fn(state, prms=prms, restraint=restraint)
        #return (state, nbpairs)
        return state, restraint, prms

    inner = 100
    outer = int(min_steps/inner)

    initial_torsions = []
    actual_torsions = []
    pre_energies = []
    pre_rest_energies = []
    energies = []
    rest_energies = []
    pairs = []
    mdtimes = []
    post_positions = []

    global iteration
    iteration = iteration + 1

    if iteration % min_interval == 0:
        print("Minimization Run")
        # vmap instead of naive loop
        target_angle = jnp.array([i for i in range(NPOINTS)])
        crds = jnp.array(crds)
        batch_inner = jax.vmap(min_inner, in_axes=(0, 0, None, None, None, None, None, None, None, None), out_axes=(0,0,0))

        energies, post_positions, actual_torsions = batch_inner(crds, target_angle, energy_fn_no_restraint, energy_fn, init_fn, body_fn, masses, boxVectors, prms, torsions)

        deviation = []
        for i, j in enumerate(actual_torsions):
            current_angle = j * radian_to_degree
            current_angle = jnp.where(current_angle < 0.0, current_angle + 360.0, current_angle)
            deviation.append(jnp.absolute(i*10 - current_angle))

        print("Average angular deviation from restrained angle:", jnp.mean(jnp.array(deviation)))
        print("Individual deviations from restrained angle:", deviation)
    else:
        post_positions = crds
        energies = jnp.array([energy_fn_no_restraint(p, prms=prms) for p in crds])

    return energies, post_positions

def min_inner(crds, target, energy_fn_no_restraint, energy_fn, init_fn, body_fn, masses, boxVectors, prms, torsions):
    radian_to_degree = 180.0/jnp.pi
    degree_to_radian = 1.0/radian_to_degree
    frc1 = 1000
    frc2 = 0.1
    t = torsions[0]

    target_angle = target * 10 * degree_to_radian
    curr_rest = [t[0],t[1],t[2],t[3], target_angle, frc1, frc2]

    current_crds = crds

    pre_energies = energy_fn_no_restraint(current_crds, prms=prms)

    state = init_fn(current_crds, mass=masses, restraint=curr_rest, prms=prms)

    state = jax_md.minimize.FireDescentState(jnp.float64(state.position),jnp.float64(state.momentum),\
                                                jnp.float64(state.force), state.mass, state.dt, state.alpha,\
                                                state.n_pos)

    p1 = state.position[t[0]]
    p2 = state.position[t[1]]
    p3 = state.position[t[2]]
    p4 = state.position[t[3]]
    initial_torsions = amber.torsion_single(p1,p2,p3,p4, boxVectors)

    iter = 2000
    inner = 100
    outer = int(iter/inner)

    for i in range(outer):
        state, curr_rest, prms = jax.lax.fori_loop(0, inner, body_fn, (state, curr_rest, prms))

    p1 = state.position[t[0]]
    p2 = state.position[t[1]]
    p3 = state.position[t[2]]
    p4 = state.position[t[3]]
    actual_torsions = amber.torsion_single(p1,p2,p3,p4, boxVectors)

    post_positions = state.position
    energies = energy_fn_no_restraint(state.position, prms=prms)

    return energies, post_positions, actual_torsions

def compute_loss(residuals, loss_type='linear', delta=1.0):
    """
    Compute loss from residuals using different loss functions.

    Parameters:
    -----------
    residuals : jnp.array
        Difference between reference and computed energies
    loss_type : str
        Loss function type: 'linear' (SSE), 'huber', 'soft_l1', 'cauchy', 'arctan'
    delta : float
        Scaling parameter for robust loss functions (default=1.0)

    Returns:
    --------
    loss : float
        Computed loss value
    """
    if loss_type == 'linear':
        # Standard sum of squared errors
        return jnp.sum(residuals ** 2)

    elif loss_type == 'huber':
        # Huber loss: quadratic for small errors, linear for large errors
        # Robust to outliers
        abs_residuals = jnp.abs(residuals)
        quadratic = jnp.where(abs_residuals <= delta,
                              0.5 * residuals ** 2,
                              0.0)
        linear = jnp.where(abs_residuals > delta,
                          delta * (abs_residuals - 0.5 * delta),
                          0.0)
        return jnp.sum(quadratic + linear)

    elif loss_type == 'soft_l1':
        # Soft L1 loss: smooth approximation of L1 loss
        return jnp.sum(2 * delta**2 * (jnp.sqrt(1 + (residuals / delta)**2) - 1))

    elif loss_type == 'cauchy':
        # Cauchy loss: very robust to outliers
        return jnp.sum(delta**2 * jnp.log(1 + (residuals / delta)**2))

    elif loss_type == 'arctan':
        # Arctan loss: bounded loss function
        return jnp.sum(delta**2 * jnp.arctan((residuals / delta)**2))

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def gradObj(scipy_params, *args):
    crds, boxVectors, ref_ene, post_positions, prms_pre, torsions, loss_type, loss_delta = args

    prms = prms_pre

    i = 0
    for idx in torsions[:, 4]:
        prms._prmtop._raw_data['DIHEDRAL_FORCE_CONSTANT'][idx] = scipy_params[i]
        prms._prmtop._raw_data['SCEE_SCALE_FACTOR'][idx] = scipy_params[i+1]
        prms._prmtop._raw_data['SCNB_SCALE_FACTOR'][idx] = scipy_params[i+2]
        i = i + 3

    bondprm = amber.bond_init(prms._prmtop)
    angleprm = amber.angle_init(prms._prmtop)
    torsionprm = amber.torsion_init(prms._prmtop)
    ljprm = amber.lj_init(prms._prmtop)
    coulprm = amber.coul_init(prms._prmtop)
    prms = (bondprm, angleprm, torsionprm, ljprm, coulprm)
    def energy_fn(pos, prms=None, restraint=None):
        bprm, aprm, tprm, lprm, cprm, = prms
        return jnp.float32((amber.bond_get_energy(pos, boxVectors, bprm) \
                + amber.angle_get_energy(pos, boxVectors, aprm) \
                + amber.torsion_get_energy(pos, boxVectors, tprm) \
                + amber.lj_get_energy(pos, boxVectors, lprm) \
                + amber.coul_get_energy(pos, boxVectors, cprm))/4.184)
                #+ amber.rest_get_energy(pos, boxVectors, restraint=restraint))/4.184)

    ene_list = [energy_fn(p, prms=prms) for p in post_positions]

    min_ene = min(ene_list)

    relative_ene_list = [(x - min_ene) for x in ene_list]

    np_relative_ene_list=jnp.array(relative_ene_list)
    np_ref_ene=jnp.array(ref_ene)

    residuals = np_ref_ene - np_relative_ene_list
    loss = compute_loss(residuals, loss_type=loss_type, delta=loss_delta)

    return loss, relative_ene_list

# updates amber prmtop file, runs constrained optimizations, computes difference between ref and computed energy profiles and RMSD.
def ObjectiveFunction(scipy_params, *args):
    global iteration
    iteration = iteration + 1
    print("Iteration:", iteration)

    crds, boxVectors, ref_ene, params_dict, optvars_dict, prms, torsions, min_steps, outdir, prmtop_dir, min_interval, crd_flist, geo_dir, amber_dir, loss_type, loss_delta, param_to_optimizer_idx = args

    print("Updated Parameters:", scipy_params)

    # Set new parameters - ONLY heights (phases/scee/scnb are fixed)
    # Use param_to_optimizer_idx mapping for coupling
    for key, value in params_dict.items():
        optimizer_idx = param_to_optimizer_idx[key]
        value['height'] = scipy_params[optimizer_idx]

    # Update parmtop file in memory with new parameters
    # Convert multi-primitive format (separate _p1, _p2, _p3 keys) to grouped format
    params_dict_grouped = ConvertMultiPrimParamsForParmEd(params_dict)
    prmtop = UpdateParmTopInMemory(prmtop_dir, params_dict_grouped)
    # Save updated prmtop to disk
    prmtop.save(prmtop_dir, overwrite=True)

    # ene_list, post_positions = constrained_minimization_vec(crds, prms, boxVectors, min_steps, torsions, min_interval)

    # TODO make sure this below works
    # if interval mod current interval = 0:
    # make sure to add this option again
    # run torsional scan calculations using geometric and sander
    #print("Iter", iteration)
    #print("cut", iteration-1 % min_interval)
    #print("cutp", iteration-1 % min_interval == 0)

    if (iteration-1) % min_interval == 0:
        ierr = os.system('cd %s && rm -rf *.tmp *.log *.out *_optim* *.restrt *.path* *.rst7 *_post.xyz' % (amber_dir))
        run_task_command="""bash run_task_scipyopt_0.sh > run_task_scipyopt_0.log 2>&1 &
        pid=$!
        echo $pid > run_task.pid
        wait $pid
        """
        ierr = os.system(run_task_command)
        if(ierr != 0):
            print('Error: Please check the run.log file.')
            return

    # extract data
    ene_list=list()
    output_flist=[geo_dir + '_%03d' % (i) + '.out' for i in range(NPOINTS)]
    #TODO: change this
    for fname in output_flist:
        with open(fname, 'r') as f:
            lines=f.readlines()

        ene=[float(re.findall(r'(\-*\d+\.\d+)',l)[0]) for l in lines if re.match(r'.*Final converged energy:.*',l)]

        if(len(ene) == 0):
            print('Error: Failed to extract energy from %s.' % (fname))
            return

        ene_list.append(ene[-1])

    crd_flist=[geo_dir + '_%03d' % (i) + '_post.xyz' for i in range(NPOINTS)]
    post_positions = extractCoordinates(crd_flist)
    #TODO make sure you compare amber energies and the ones generated by this

    loss_and_grad_fn = jax.value_and_grad(gradObj, has_aux=True)
    loss_and_grad = loss_and_grad_fn(scipy_params, crds, boxVectors, ref_ene, post_positions, prms, torsions, loss_type, loss_delta)

    #global iteration
    #iteration = iteration + 1
    #print("Extracted energy list:", ene_list)

    loss_ene, grad = loss_and_grad
    loss, jax_ene_list = loss_ene
    grad = grad.astype('float64')
    jax_energies = [float(val) for val in jax_ene_list]
    print("JAX Energies:", jax_energies) # AB: Save JAX energy for each iteration. 
    print("Loss", loss)
    losses.append(loss)
    print("Loss Grad", grad)

    #TODO test SSE against RMSD
    #also check internal jax energies against this
    #ie print(ene_list, ene_list_jax) and diff
    #ene_list_jax = [energy_fn(p, prms=prms) for p in post_positions]
    min_ene = min(ene_list)
    relative_ene_list = [(x - min_ene) * 627.5 for x in ene_list]

    global gaff_ene_list
    if iteration == 1:
        gaff_ene_list = relative_ene_list

    dihedral_label = os.path.basename(os.path.normpath(geo_dir)) # AB: Include dih label in the plot title.
    
    # AB: changed color and labels of the plots. 
    step = int(360/NPOINTS)
    plt.plot(range(0,360,step), ref_ene, marker='o', color='k', label="Reference")
    #plt.plot(range(0,360,step), relative_ene_list, marker='o', label="Sander Energies Post Optimization")
    plt.plot(range(0,360,step), gaff_ene_list, marker='o', color='r', label="Standard GAFF2")
    plt.plot(range(0,360,step), jax_ene_list, marker='o', color='b', label="AFFDO GAFF2")
    plt.title("JAX-AMBER + DLFind Fitting Iteration %s - %s" % (iteration, dihedral_label))
    plt.xlabel("Dihedral (degree)")
    plt.ylabel("Energy (kcal/mol)")
    plt.legend()
    plt.savefig(outdir + "/iteration_%s.png" % iteration)
    plt.close()

    # AB: Copy the entire geo_dir to preserve output files for each iteration.
    import shutil
    iteration_folder = os.path.join(outdir, f"iteration_{iteration}")
    shutil.copytree(os.path.dirname(geo_dir), iteration_folder, dirs_exist_ok=True)

    # Update best values if applicable
    global best_loss
    global best_params
    global best_iteration
    global best_energy
    if loss < best_loss:
        best_loss = loss
        best_params = scipy_params
        best_iteration = iteration
        best_energy = jax_energies  # AB: Save the energy profile for the best iteration

    # scipy requires jac gradient as list
    # print("loss", loss)
    # print("grad", grad)
    #sys.exit()
    return loss, list(grad)

def GetLinearGuess(params_dict_single, ref_ene, conf_lbl, dh_lbl, nprim, torsion_coupling_mode='semi-independent', geo_dir=None, prmtop_dir=None):
    """
    Get linear least-squares initial guess (FFPOpt-style) for multi-primitive heights.

    NEW ADVANCED ALGORITHM (mirrors SciPy BuildLinearGuessScript):
    1. Compute actual dihedral angles from XYZ geometries
    2. Compute low-level energies (MM WITHOUT torsions being fitted)
    3. Fit against TRUE torsion contribution: residual = ref_ene - llenes
    4. Support torsion coupling (fully-independent, semi-independent, fully-coupled)
    5. Use pseudoinverse with constant term for robust solving

    Returns multi-primitive params dict with optimized heights.
    """
    import numpy as np
    import os, glob
    from openmm import app

    print("\n" + "="*60)
    print("LINEAR LEAST-SQUARES INITIAL GUESS (FFPOpt-style)")
    print("="*60)

    # Convert reference energies to numpy
    ref_ene_array = np.array(ref_ene, dtype=float)
    npoints = len(ref_ene_array)

    # Check if we have geometries and prmtop for advanced algorithm
    use_advanced = geo_dir is not None and prmtop_dir is not None

    if not use_advanced:
        print("  Using simple algorithm (no geometry dir provided)")
        print(f"  Assuming uniform torsion scan: {npoints} points")

        # Simple algorithm: assume uniform scan from 0 to 360 degrees
        angles = np.linspace(0, 2*np.pi, npoints, endpoint=False)

        # Build design matrix for multi-primitive fit
        # E(θ) = Σ V_n/2 * (1 + cos(n*θ))  for n=1,2,3,...
        # Note: AMBER uses V_n/2, not V_n
        X = np.zeros((npoints, nprim))
        for i in range(nprim):
            n = i + 1  # periodicity 1, 2, 3, ...
            X[:, i] = 0.5 * (1.0 + np.cos(n * angles))

        # Solve linear system: X @ heights = ref_ene
        heights, residuals, rank, s = np.linalg.lstsq(X, ref_ene_array, rcond=None)

        # Compute RMSD
        fitted_ene = X @ heights
        rmsd = np.sqrt(np.mean((ref_ene_array - fitted_ene)**2))

        print(f"  nprim={nprim}, npoints={npoints}")
        print(f"  RMSD: {rmsd:.4f} kcal/mol")

        # Create multi-primitive params dict
        multi_prim_dict = {}
        for torsion_key, params in params_dict_single.items():
            for i in range(nprim):
                n = i + 1
                new_key = f"{torsion_key}_p{n}"
                # Allow negative heights (physical in multi-primitive fits)
                multi_prim_dict[new_key] = {
                    'height': float(heights[i]),
                    'phase': 0.0,
                    'periodicity': n,
                    'scee': params.get('scee', 1.2),
                    'scnb': params.get('scnb', 2.0),
                    'torsion_mask': params.get('torsion_mask', [])
                }
                print(f"    {new_key}: height={heights[i]:.4f}, phase=0.0°, periodicity={n}")

        return multi_prim_dict

    # ADVANCED ALGORITHM: Use actual geometries and compute low-level energies
    print("  Using advanced algorithm (with geometry analysis)")
    print(f"  Loaded {npoints} reference energies")

    # Load XYZ files
    xyz_pattern = os.path.join(geo_dir, '*.xyz')
    xyz_files = sorted(glob.glob(xyz_pattern))

    if len(xyz_files) != npoints:
        print(f"  Warning: Found {len(xyz_files)} geometries but {npoints} energies")
        print(f"  Trimming to {min(len(xyz_files), npoints)} points")
        trim_to = min(len(xyz_files), npoints)
        xyz_files = xyz_files[:trim_to]
        ref_ene_array = ref_ene_array[:trim_to]
        npoints = trim_to

    # Parse atom names from prmtop
    def parse_atom_names(prmtop_path):
        names = []
        with open(prmtop_path, 'r') as fh:
            reading = False
            for line in fh:
                stripped = line.strip()
                if stripped.startswith('%FLAG ATOM_NAME'):
                    reading = True
                    continue
                if not reading:
                    continue
                if stripped.startswith('%FLAG ') and not stripped.startswith('%FLAG ATOM_NAME'):
                    break
                if stripped.startswith('%FORMAT'):
                    continue
                if not stripped:
                    continue
                for i in range(0, len(line), 4):
                    token = line[i:i+4].strip()
                    if token:
                        names.append(token)
        return names

    atom_names = parse_atom_names(prmtop_dir)
    print(f"  Loaded {len(atom_names)} atoms from prmtop")

    # Load XYZ coordinates
    def load_xyz(path):
        with open(path, 'r') as fh:
            lines = fh.readlines()
        natoms = int(lines[0].split()[0])
        coords = []
        for line in lines[2:2+natoms]:
            parts = line.split()
            if len(parts) >= 4:
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(coords, dtype=float)

    coords_stack = [load_xyz(path) for path in xyz_files]
    natoms = coords_stack[0].shape[0]
    print(f"  Loaded {len(coords_stack)} conformers ({natoms} atoms each)")

    # Compute dihedral angles from geometries
    def compute_dihedral(coords, idx_tuple):
        p0, p1, p2, p3 = (coords[i] for i in idx_tuple)
        b0 = p0 - p1
        b1 = p2 - p1
        b2 = p3 - p2
        norm_b1 = np.linalg.norm(b1)
        if norm_b1 == 0.0:
            return 0.0
        b1_unit = b1 / norm_b1
        v = b0 - np.dot(b0, b1_unit) * b1_unit
        w = b2 - np.dot(b2, b1_unit) * b1_unit
        x = np.dot(v, w)
        y = np.dot(np.cross(b1_unit, v), w)
        return np.degrees(np.arctan2(y, x))

    # Get torsion indices
    torsion_order = list(params_dict_single.keys())
    torsion_indices = {}
    torsion_angles = {}

    for torsion_key in torsion_order:
        mask = params_dict_single[torsion_key].get('torsion_mask', [])
        if len(mask) != 4:
            raise ValueError(f'Torsion {torsion_key} has invalid torsion_mask: {mask}')
        indices = []
        for atom_name in mask:
            if atom_name not in atom_names:
                raise ValueError(f'Atom name {atom_name} not found in prmtop for torsion {torsion_key}')
            indices.append(atom_names.index(atom_name))
        torsion_indices[torsion_key] = tuple(indices)

        # Compute angles for this torsion across all conformers
        angle_list = [compute_dihedral(coords, torsion_indices[torsion_key]) for coords in coords_stack]
        torsion_angles[torsion_key] = np.array(angle_list, dtype=float)
        print(f'    {torsion_key}: angle span {np.min(angle_list):.2f}° to {np.max(angle_list):.2f}° (range {np.ptp(angle_list):.2f}°)')

    # Compute low-level energies (MM WITHOUT torsions being fitted)
    # This is the FFPOpt approach: isolates TRUE torsion contribution
    print("\n" + "="*60)
    print("COMPUTING LOW-LEVEL ENERGIES (MM without torsions)")
    print("="*60)

    # Create prmtop without torsions being fitted
    from parmedmod import CreatePrmtopWithoutTorsions

    all_torsion_masks = [params_dict_single[key]['torsion_mask'] for key in torsion_order]
    prmtop_no_torsion = prmtop_dir + '_no_torsion'

    print(f"  Creating temporary prmtop with {len(torsion_order)} torsion(s) deleted...")
    CreatePrmtopWithoutTorsions(prmtop_dir, prmtop_no_torsion, all_torsion_masks, debug=False)
    print(f"  Created: {prmtop_no_torsion}")

    # Compute energies using OpenMM
    print(f"\n  Running OpenMM single-point energies for {npoints} geometries...")

    # Load prmtop without torsions
    prmtop_no_tors = app.AmberPrmtopFile(prmtop_no_torsion)
    system_no_tors = prmtop_no_tors.createSystem(
        nonbondedMethod=app.NoCutoff,
        constraints=None
    )

    # Create integrator and context (for energy evaluation only)
    integrator_no_tors = openmm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 0.002*unit.picosecond)
    context_no_tors = openmm.Context(system_no_tors, integrator_no_tors)

    llenes = []
    for i, coords in enumerate(coords_stack):
        # Set positions (coords are in Angstroms, OpenMM uses nanometers)
        positions = coords * 0.1  # Angstrom to nm
        context_no_tors.setPositions(positions)

        # Get potential energy
        state = context_no_tors.getState(getEnergy=True)
        energy_kj = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        energy_kcal = energy_kj * 0.239006  # kJ/mol to kcal/mol

        llenes.append(energy_kcal)
        if (i + 1) % 5 == 0 or (i + 1) == npoints:
            print(f'    Geometry {i+1}/{npoints}: llene = {energy_kcal:.6f} kcal/mol')

    del context_no_tors
    del integrator_no_tors
    llenes = np.array(llenes, dtype=float)

    # Clean up temporary prmtop
    os.remove(prmtop_no_torsion)

    # Shift both llenes and ref_ene to zero minimum (FFPOpt approach)
    llenes -= np.min(llenes)
    ref_ene_shifted = ref_ene_array - np.min(ref_ene_array)

    # Compute TRUE torsion contribution: residual = ref_ene - llenes
    residual = ref_ene_shifted - llenes

    print(f"\n  Low-level energies computed successfully")
    print(f"    llenes range: {np.min(llenes):.6f} to {np.max(llenes):.6f} kcal/mol")
    print(f"    ref_ene range: {np.min(ref_ene_shifted):.6f} to {np.max(ref_ene_shifted):.6f} kcal/mol")
    print(f"    Torsion contribution range: {np.min(residual):.6f} to {np.max(residual):.6f} kcal/mol")

    # Build torsion groups based on coupling mode
    if coupling_mode == 'fully-coupled':
        torsion_groups = [torsion_order]
    elif coupling_mode == 'fully-independent':
        torsion_groups = [[key] for key in torsion_order]
    else:  # semi-independent
        pattern_groups = {}
        for key in torsion_order:
            pattern = tuple(params_dict_single[key].get('torsion_mask', []))
            pattern_groups.setdefault(pattern, []).append(key)
        torsion_groups = list(pattern_groups.values())

    # Initialize with GAFF fallback
    linear_params = {}
    for torsion_key in torsion_order:
        torsion_vals = params_dict_single[torsion_key]
        height_val = torsion_vals.get('height', 0.0)
        scee_val = torsion_vals.get('scee', 1.2)
        scnb_val = torsion_vals.get('scnb', 2.0)
        fallback_heights = [float(height_val)] + [0.0] * (nprim - 1)
        linear_params[torsion_key] = {
            'height': fallback_heights,
            'phase': [0.0] * nprim,
            'periodicity': list(range(1, nprim + 1)),
            'scee': scee_val,
            'scnb': scnb_val,
            'torsion_mask': torsion_vals.get('torsion_mask', [])
        }

    print("\n" + "="*60)
    print("MULTI-PRIMITIVE PARAMETERS (LINEAR LSQ)")
    print("="*60)

    # Fit each group
    for group in torsion_groups:
        if not group:
            continue

        # Build design matrix for this group
        group_columns = []
        for torsion_key in group:
            ang_rad = np.deg2rad(torsion_angles[torsion_key])
            for harmonic in range(1, nprim + 1):
                # AMBER convention: E = V_n/2 * (1 + cos(n*phi - gamma))
                # With phase=0: E = V_n/2 * (1 + cos(n*phi))
                group_columns.append(0.5 * (1.0 + np.cos(harmonic * ang_rad)))

        # Add constant term column (absorbs baseline offset)
        design_matrix = np.column_stack(group_columns + [np.ones(npoints)])
        target = residual.copy()

        # Use pseudoinverse for robust solving (more stable than lstsq)
        coeffs_with_const = np.linalg.pinv(design_matrix).dot(target)

        # Extract constant term
        const = coeffs_with_const[-1]
        coeffs = coeffs_with_const[:-1]  # Torsion heights (V_n parameters)

        # Compute fit quality
        fit = design_matrix[:, :-1].dot(coeffs) + const
        ss_res = float(np.sum((target - fit) ** 2))
        ss_tot = float(np.sum((target - np.mean(target)) ** 2))
        rmse = float(np.sqrt(np.mean((target - fit) ** 2)))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        if r2 < 0.0:
            print(f"  Warning: group {group} produced poor fit (RMSE={rmse:.4f}, R2={r2:.4f}); using GAFF fallback")
            continue

        # Update residual
        residual -= fit

        # Extract heights for each torsion in group
        coeff_index = 0
        for torsion_key in group:
            heights = []
            for _ in range(nprim):
                heights.append(float(coeffs[coeff_index]))
                coeff_index += 1

            torsion_vals = params_dict_single[torsion_key]
            scee_val = torsion_vals.get('scee', 1.2)
            scnb_val = torsion_vals.get('scnb', 2.0)

            linear_params[torsion_key] = {
                'height': heights,
                'phase': [0.0] * nprim,
                'periodicity': list(range(1, nprim + 1)),
                'scee': scee_val,
                'scnb': scnb_val,
                'torsion_mask': torsion_vals.get('torsion_mask', [])
            }

            heights_fmt = ', '.join('%.4f' % h for h in heights)
            print(f'    {torsion_key}: heights=[{heights_fmt}] (RMSE={rmse:.4f}, R2={r2:.4f})')

    # Convert to JAX format (separate _pN entries)
    multi_prim_dict = {}
    for torsion_key, params in linear_params.items():
        for i in range(nprim):
            n = i + 1
            new_key = f"{torsion_key}_p{n}"
            multi_prim_dict[new_key] = {
                'height': params['height'][i],  # Allow negative (physical in multi-prim)
                'phase': params['phase'][i],
                'periodicity': params['periodicity'][i],
                'scee': params['scee'],
                'scnb': params['scnb'],
                'torsion_mask': params['torsion_mask']
            }

    print("\nLinear guess generation complete!")
    return multi_prim_dict

def GetFourierGuess(params_dict_single, ref_ene):
    """
    Get Fourier-based initial guess for single-primitive parameters.

    Returns single-primitive params dict with Fourier-derived height and phase.
    """
    import numpy as np

    ref_ene_array = np.array(ref_ene, dtype=float)
    N = len(ref_ene_array)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    mean_ene = np.mean(ref_ene_array)

    # Multi-harmonic Fourier fit
    max_harmonics = 3

    # Build design matrix
    X = np.ones((len(angles), 1))  # c0 term
    for n in range(1, max_harmonics + 1):
        X = np.column_stack([X, np.cos(n * angles), np.sin(n * angles)])

    # Solve linear system
    coeffs, residuals, rank, s = np.linalg.lstsq(X, ref_ene_array, rcond=None)

    # Reconstruct and compute R2
    reconstruction = X @ coeffs
    ss_res = np.sum((ref_ene_array - reconstruction)**2)
    ss_tot = np.sum((ref_ene_array - mean_ene)**2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    rmse = np.sqrt(ss_res / len(ref_ene_array))

    print(f"\nFourier-based initial guess:")
    print(f"  R²: {r2:.4f}, RMSE: {rmse:.4f} kcal/mol")

    # Extract harmonics
    harmonics = []
    for i in range(max_harmonics):
        a_n = coeffs[1 + 2*i]
        b_n = coeffs[1 + 2*i + 1]
        A = np.sqrt(a_n**2 + b_n**2)
        phi = np.arctan2(b_n, a_n)
        harmonics.append((i+1, A, phi))
        print(f"    n={i+1}: A={A:.4f}, phi={np.degrees(phi):7.1f}°")

    # Select best harmonic (prefer n=2 or n=3)
    max_amplitude = max(h[1] for h in harmonics)
    threshold = 0.8 * max_amplitude

    preferred = [(n, A, phi) for n, A, phi in harmonics if n in [2, 3] and A >= threshold]
    if preferred:
        periodicity_0, A_0, phi_0 = max(preferred, key=lambda x: x[1])
    else:
        periodicity_0, A_0, phi_0 = max(harmonics, key=lambda x: x[1])

    # Wrap phase to [0°, 180°]
    phi_0_deg = np.degrees(phi_0) % 180.0

    print(f"  Selected: n={periodicity_0}, A={A_0:.4f}, phase={phi_0_deg:.1f}°")

    # Create single-primitive guess with Fourier height and phase
    fourier_params = {}
    for torsion_key, params in params_dict_single.items():
        fourier_params[torsion_key] = params.copy()
        fourier_params[torsion_key]['height'] = A_0
        fourier_params[torsion_key]['phase'] = phi_0_deg

    return fourier_params

def GetStandardGuess(params_dict_single):
    """
    Get standard GAFF initial guess (use params as-is).
    """
    print("\nUsing standard GAFF initial guess from params.json")
    return params_dict_single.copy()

def ConvertToMultiPrimitive(params_dict, nprim=3, phase_strategy='fixed_zero'):
    """
    Convert single-primitive torsion parameters to multi-primitive.

    Takes a dictionary with single torsion entries like:
        {'7-9-10-11': {'height': 2.5, 'phase': 180.0, 'periodicity': 2, ...}}

    Returns a dictionary with multiple periodicities:
        {'7-9-10-11_p1': {'height': h1, 'phase': 0.0, 'periodicity': 1, ...},
         '7-9-10-11_p2': {'height': h2, 'phase': 0.0, 'periodicity': 2, ...},
         '7-9-10-11_p3': {'height': h3, 'phase': 0.0, 'periodicity': 3, ...}}

    Parameters:
    -----------
    params_dict : dict
        Single-primitive parameters
    nprim : int
        Number of primitives (3 or 6)
    phase_strategy : str
        'fixed_zero' or 'alternating'
    """
    multi_prim_dict = {}

    # Define phases based on strategy
    if phase_strategy == 'fixed_zero':
        phases = [0.0] * nprim
    elif phase_strategy == 'alternating':
        phases = [0.0, 180.0, 0.0, 180.0, 0.0, 0.0][:nprim]
    else:
        raise ValueError(f"Unknown phase_strategy: {phase_strategy}")

    # Define periodicities
    periodicities = list(range(1, nprim + 1))

    for torsion_key, params in params_dict.items():
        # Get initial height from single primitive
        initial_height = params.get('height', 1.0)

        # Create nprim entries for this torsion
        for i, (periodicity, phase) in enumerate(zip(periodicities, phases), start=1):
            new_key = f"{torsion_key}_p{periodicity}"
            multi_prim_dict[new_key] = {
                'height': initial_height / nprim,  # Distribute initial height
                'phase': phase,
                'periodicity': periodicity,
                'scee': params.get('scee', 1.2),  # Keep original scaling
                'scnb': params.get('scnb', 2.0),
                'torsion_mask': params.get('torsion_mask', [])  # Preserve torsion_mask
            }

    return multi_prim_dict

def GroupTorsionsByAtomType(params_dict, prmtop_file):
    """
    Group torsions by their atom type pattern.

    This allows semi-independent fitting where torsions with the same atom type
    pattern (e.g., c3-c3-c3-c3) share parameters, but different patterns
    (e.g., c3-c3-c3-hc) have independent parameters.

    Args:
        params_dict: Dictionary of torsion parameters with 'torsion_mask' for each
        prmtop_file: Path to prmtop file to extract atom types

    Returns:
        dict: {type_pattern: [torsion_key1, torsion_key2, ...]}
              type_pattern is like 'c3-c3-c3-c3'
    """
    import parmed as pmd

    # Load prmtop to get atom types
    try:
        parm = pmd.load_file(prmtop_file)
    except Exception as e:
        print(f"Warning: Could not load prmtop file '{prmtop_file}': {e}")
        print("  Falling back to fully-coupled mode (all torsions in one group)")
        # Return all torsions in a single group if we can't load prmtop
        return {'unknown-type': list(params_dict.keys())}

    # Build mapping from atom name to atom type
    name_to_type = {atom.name: atom.type for atom in parm.atoms}

    # Group torsions by atom type pattern
    type_groups = {}

    for torsion_key, torsion_data in params_dict.items():
        # For multi-primitive keys like "7-9-10-11_p1", extract base key
        base_key = torsion_key.split('_p')[0] if '_p' in torsion_key else torsion_key

        # Get atom names for this torsion
        atom_names = torsion_data.get('torsion_mask', [])

        if len(atom_names) != 4:
            print(f"Warning: Torsion {torsion_key} has invalid torsion_mask: {atom_names}")
            continue

        # Get atom types for these names
        try:
            atom_types = [name_to_type[name] for name in atom_names]
            type_pattern = '-'.join(atom_types)
        except KeyError as e:
            print(f"Warning: Atom name {e} not found in prmtop for torsion {torsion_key}")
            continue

        # Add to group
        if type_pattern not in type_groups:
            type_groups[type_pattern] = []
        type_groups[type_pattern].append(torsion_key)

    return type_groups

def ff_opt(prmtop_dir, params_dir, geo_dir, amber_dir, min_steps, opt_loops, ref_ene, outdir, min_interval, nprim=3, multi_prim_phase_strategy='fixed_zero', initial_guess_method='linear', auto_nprim_fallback=True, nprim_fallback_threshold=1.0, loss='huber', loss_delta=1.0, torsion_coupling_mode='semi-independent'):
    """
    Force field optimization using JAX with multi-primitive strategy.

    Parameters:
    -----------
    nprim : int, default=3
        Number of primitives (periodicities) to use:
        - nprim=1: Single primitive (original behavior, not recommended)
        - nprim=3: Multi-primitive with [1,2,3] periodicities (like ffpopt)
        - nprim=6: Extended multi-primitive with [1,2,3,4,5,6] periodicities

    multi_prim_phase_strategy : str, default='fixed_zero'
        How to handle phases in multi-primitive mode:
        - 'fixed_zero': Fix all phases to 0° (like ffpopt) - RECOMMENDED
        - 'alternating': Alternate between 0° and 180° for periodicities [1,2,3,4,5,6]

    initial_guess_method : str, default='linear'
        Method for generating initial guess:
        - 'linear': Linear LSQ guess (FFPOpt-style) - RECOMMENDED for multi-primitive
        - 'fourier': Fourier-based guess (good for complex profiles)
        - 'gaff': Standard GAFF guess from params.json

    auto_nprim_fallback : bool, default=True
        Automatically fallback to nprim=6 if nprim=3 doesn't achieve good fit.
        If True and best loss corresponds to RMSD > nprim_fallback_threshold,
        retry with nprim=6.

    nprim_fallback_threshold : float, default=0.5
        RMSD threshold (kcal/mol) that triggers automatic fallback to nprim=6.
        Only used if auto_nprim_fallback=True and nprim=3.

    loss : str, default='huber'
        Loss function type:
        - 'linear': Sum of squared errors (SSE) - standard but sensitive to outliers
        - 'huber': Huber loss - robust to outliers (RECOMMENDED)
        - 'soft_l1': Soft L1 loss - smooth approximation of L1
        - 'cauchy': Cauchy loss - very robust to outliers
        - 'arctan': Arctan loss - bounded loss function

    loss_delta : float, default=1.0
        Scaling parameter for robust loss functions (used in huber, soft_l1, cauchy, arctan).
        Controls transition point between quadratic and linear behavior.

    torsion_coupling_mode : str, default='semi-independent'
        How to couple torsions across the molecule:
        - 'semi-independent': Group torsions by atom-type pattern (RECOMMENDED, default)
        - 'fully-independent': Each torsion has independent parameter set
        - 'fully-coupled': Single parameter set shared by all torsions

    Note: Only barrier heights are optimized. Phases are FIXED based on strategy.
          scee and scnb scaling factors are NOT optimized (kept at default values).
    """
    # Declare global variables that track best optimization results
    global best_loss, best_params, best_iteration, best_energy, losses, iteration

    initial_guess='initial_guess'
    algorithm='L-BFGS-B'
    # maxiter=1000
    step_size=0.100000

    print("="*60)
    print("JAX MULTI-PRIMITIVE TORSION OPTIMIZER")
    print("="*60)
    print(f"nprim                 : {nprim}")
    print(f"phase_strategy        : {multi_prim_phase_strategy}")
    print(f"initial_guess_method  : {initial_guess_method}")
    print(f"loss_function         : {loss} (delta={loss_delta})")
    print(f"torsion_coupling      : {torsion_coupling_mode}")
    if auto_nprim_fallback and nprim == 3:
        print(f"auto_nprim_fallback   : Enabled (threshold={nprim_fallback_threshold} kcal/mol)")
    print("="*60)

    crd_flist=[geo_dir + '_%03d' % (i) + '.xyz' for i in range(NPOINTS)]

    #list of 36 (35,3) numpy arrays from 0-350 deg
    coordinates = extractCoordinates(crd_flist)

    # Load single-primitive parameters
    params_dict_single=ReadJsonData(params_dir)[initial_guess]
    optvars_dict=ReadJsonData(params_dir)['optvars']
    bounds_dict=ReadJsonData(params_dir)['bounds']

    # Load reference energies
    ref_ene_data = ReadJsonData(ref_ene)['ref_ene']
    conf_lbl_str = os.path.basename(os.path.dirname(geo_dir))
    dh_lbl_str = os.path.basename(geo_dir)

    # Select initial guess method
    if initial_guess_method == 'linear':
        # Linear LSQ directly generates multi-primitive parameters
        print(f"\nUsing Linear LSQ initial guess (FFPOpt-style)")
        params_dict = GetLinearGuess(params_dict_single, ref_ene_data, conf_lbl_str, dh_lbl_str, nprim,
                                      torsion_coupling_mode=torsion_coupling_mode,
                                      geo_dir=geo_dir, prmtop_dir=prmtop_dir)
    elif initial_guess_method == 'fourier':
        # Fourier generates single-primitive, then convert to multi-primitive
        print(f"\nUsing Fourier initial guess")
        params_dict_fourier = GetFourierGuess(params_dict_single, ref_ene_data)
        if nprim > 1:
            print(f"Converting Fourier guess to multi-primitive (nprim={nprim})")
            params_dict = ConvertToMultiPrimitive(params_dict_fourier, nprim=nprim, phase_strategy=multi_prim_phase_strategy)
        else:
            params_dict = params_dict_fourier
    elif initial_guess_method == 'gaff':
        # Standard GAFF, then convert to multi-primitive
        print(f"\nUsing standard GAFF initial guess")
        params_dict_gaff = GetStandardGuess(params_dict_single)
        if nprim > 1:
            print(f"Converting GAFF guess to multi-primitive (nprim={nprim})")
            params_dict = ConvertToMultiPrimitive(params_dict_gaff, nprim=nprim, phase_strategy=multi_prim_phase_strategy)
        else:
            params_dict = params_dict_gaff
    else:
        raise ValueError(f"Unknown initial_guess_method: {initial_guess_method}")

    # Build initial guess and bounds based on coupling mode - ONLY for heights
    guess = list()
    bounds = list()

    # Create mapping from parameter to optimizer index for coupling
    param_to_optimizer_idx = {}  # Maps each torsion key to its optimizer index

    if torsion_coupling_mode == 'fully-independent':
        # FULLY INDEPENDENT: Each torsion has its own parameters
        print(f"\nBuilding guess/bounds for FULLY-INDEPENDENT mode ({len(params_dict)} parameters)")
        for idx, (key, value) in enumerate(params_dict.items()):
            guess.append(value['height'])
            bounds.append(bounds_dict['height'])
            param_to_optimizer_idx[key] = idx

    elif torsion_coupling_mode == 'semi-independent':
        # SEMI-INDEPENDENT: Group by atom type pattern, share within group
        type_groups = GroupTorsionsByAtomType(params_dict, prmtop_dir)
        print(f"\nBuilding guess/bounds for SEMI-INDEPENDENT mode ({len(type_groups)} atom type groups)")

        idx = 0
        for type_pattern, torsion_keys in type_groups.items():
            print(f"  Group '{type_pattern}': {len(torsion_keys)} torsions")
            # Use first torsion in group as representative
            representative_key = torsion_keys[0]
            guess.append(params_dict[representative_key]['height'])
            bounds.append(bounds_dict['height'])

            # Map all torsions in this group to same optimizer index
            for torsion_key in torsion_keys:
                param_to_optimizer_idx[torsion_key] = idx
            idx += 1

    elif torsion_coupling_mode == 'fully-coupled':
        # FULLY COUPLED: All torsions share the same parameters
        print(f"\nBuilding guess/bounds for FULLY-COUPLED mode (1 parameter set for {len(params_dict)} torsions)")
        # Use first torsion as representative
        first_key = list(params_dict.keys())[0]
        guess.append(params_dict[first_key]['height'])
        bounds.append(bounds_dict['height'])

        # Map all torsions to index 0
        for key in params_dict.keys():
            param_to_optimizer_idx[key] = 0
    else:
        raise ValueError(f"Unknown torsion_coupling_mode: {torsion_coupling_mode}")

    print(f"Number of parameters to optimize: {len(guess)} (heights only)")
    print(f"Initial guess: {guess}")

    # Make initial FF modifications using parmed in-memory
    rng = np.random.default_rng()
    for k in params_dict:
        params_dict[k]['height'] += rng.random() # * 1e-5 too small of a value and parmed will truncate it, adjust this if you'd like
    # Convert multi-primitive format to grouped format for ParmEd
    params_dict_grouped_initial = ConvertMultiPrimParamsForParmEd(params_dict)
    prmtop_initial = UpdateParmTopInMemory(prmtop_dir, params_dict_grouped_initial)
    prmtop_initial.save(prmtop_dir, overwrite=True)

    prmtopomm = app.AmberPrmtopFile(prmtop_dir)

    # Grab all indices of torsions from the params file
    # For multi-primitive, keys are like "7-9-10-11_p1", "7-9-10-11_p2", etc.
    # Extract unique base torsions (remove _pN suffix)
    unique_torsions = set()
    for key in params_dict.keys():
        # Remove _pN suffix if present
        base_key = key.split('_p')[0] if '_p' in key else key
        unique_torsions.add(base_key)

    torsions = [list(map(int, torsion.split("-"))) for torsion in unique_torsions]
    ref_ene = jnp.array(ReadJsonData(ref_ene)['ref_ene'])
    print("Torsion Indices from parameter file:", torsions)
    print(f"Number of unique torsions: {len(torsions)}, Number of parameter entries: {len(params_dict)}")

    # Use regular numpy to prevent tracing to make this easier
    torsionidx = prmtopomm._prmtop._raw_data["DIHEDRALS_INC_HYDROGEN"] + prmtopomm._prmtop._raw_data["DIHEDRALS_WITHOUT_HYDROGEN"]
    torsionidx = np.array([int(index) for index in torsionidx]).reshape((-1,5))
    torsionidx[:, :4] = torsionidx[:, :4]//3
    torsionidx[:, 4] = torsionidx[:, 4]-1
    print("All Torsion Indices:", torsionidx)

    # Find the actual parameter index in the prmtop file using the atom numbers for the torsion
    torsion_indices = []
    torsion_idx_list = torsionidx.tolist()
    for torsion in torsions:
        for torsion_idx in torsion_idx_list:
            if torsion == torsion_idx[:4]:
                torsion_indices.append(torsion_idx)

    torsion_indices = jnp.array(torsion_indices)

    print("Selected Torsion & Parameter Indices:")
    print(torsion_indices)

    print("Torsion to be constrained:", torsions[0][:4])

    # sys.exit()

    #system = prmtopomm.createSystem(nonbondedMethod=app.NoCutoff, removeCMMotion=False, constraints=None)
    #boxVectors = jnp.array([v._value for v in system.getDefaultPeriodicBoxVectors()])
    #boxVectors = boxVectors.sum(axis=0)
    boxVectors = jnp.array([100.0, 100.0, 100.0])

    minimization_result=minimize(ObjectiveFunction, guess, jac=True, \
           args=(coordinates, boxVectors, ref_ene, params_dict, optvars_dict, prmtopomm, torsion_indices,
                 min_steps, outdir, prmtop_dir, min_interval, crd_flist, geo_dir, amber_dir, loss, loss_delta, param_to_optimizer_idx), \
           bounds=bounds, method=algorithm, options={'maxiter':opt_loops, 'eps': step_size})

    print("Losses:", losses)

    print("Best Loss:", best_loss)

    print("Best Iteration:", best_iteration)

    print("Best Energies:", best_energy) # AB: Print the energy profile corresponding to the best iteration. 
    
    print("Best Params:", best_params)

    print("Final Params: ", minimization_result.x)

    print("Termination Message: ", minimization_result.message)

    # Check if auto-fallback to 6 primitives is needed
    if auto_nprim_fallback and nprim == 3:
        # Store initial nprim result (from global variables)
        initial_nprim = nprim
        best_loss_3prim = float(best_loss)
        best_params_3prim = best_params.copy() if best_params is not None else None
        best_energy_3prim = best_energy.copy() if best_energy is not None else None
        best_iteration_3prim = int(best_iteration)

        # Compute RMSD from best loss (SSE)
        ref_ene_array = jnp.array(ref_ene_data)
        best_rmsd_3prim = jnp.sqrt(best_loss_3prim / len(ref_ene_array))

        print("\n" + "="*60)
        print("AUTO-FALLBACK CHECK (nprim=3 → nprim=6)")
        print("="*60)
        print(f"Best RMSD with 3 primitives: {best_rmsd_3prim:.6f} kcal/mol")
        print(f"Threshold for fallback     : {nprim_fallback_threshold:.6f} kcal/mol")

        if best_rmsd_3prim > nprim_fallback_threshold:
            print(f"\n→ RMSD exceeds threshold, falling back to 6 primitives")
            print("="*60)

            # Reset global trackers for 6-prim optimization
            best_loss = jnp.iinfo(jnp.int64).max
            best_params = None
            best_iteration = -1
            best_energy = None
            losses = []
            iteration = 0

            print("\nRegenerating initial guess with 6 primitives...")

            # Regenerate guess with nprim=6
            if initial_guess_method == 'linear':
                params_dict_6prim = GetLinearGuess(params_dict_single, ref_ene_data, conf_lbl_str, dh_lbl_str, 6,
                                                    torsion_coupling_mode=torsion_coupling_mode,
                                                    geo_dir=geo_dir, prmtop_dir=prmtop_dir)
            elif initial_guess_method == 'fourier':
                params_dict_fourier_6 = GetFourierGuess(params_dict_single, ref_ene_data)
                params_dict_6prim = ConvertToMultiPrimitive(params_dict_fourier_6, nprim=6, phase_strategy=multi_prim_phase_strategy)
            elif initial_guess_method == 'gaff':
                params_dict_gaff_6 = GetStandardGuess(params_dict_single)
                params_dict_6prim = ConvertToMultiPrimitive(params_dict_gaff_6, nprim=6, phase_strategy=multi_prim_phase_strategy)

            # Build new guess and bounds for 6 primitives with coupling
            guess_6prim = []
            bounds_6prim = []
            param_to_optimizer_idx_6prim = {}

            if torsion_coupling_mode == 'fully-independent':
                for idx, (key, value) in enumerate(params_dict_6prim.items()):
                    guess_6prim.append(value['height'])
                    bounds_6prim.append(bounds_dict['height'])
                    param_to_optimizer_idx_6prim[key] = idx

            elif torsion_coupling_mode == 'semi-independent':
                type_groups_6prim = GroupTorsionsByAtomType(params_dict_6prim, prmtop_dir)
                idx = 0
                for type_pattern, torsion_keys in type_groups_6prim.items():
                    representative_key = torsion_keys[0]
                    guess_6prim.append(params_dict_6prim[representative_key]['height'])
                    bounds_6prim.append(bounds_dict['height'])
                    for torsion_key in torsion_keys:
                        param_to_optimizer_idx_6prim[torsion_key] = idx
                    idx += 1

            elif torsion_coupling_mode == 'fully-coupled':
                first_key = list(params_dict_6prim.keys())[0]
                guess_6prim.append(params_dict_6prim[first_key]['height'])
                bounds_6prim.append(bounds_dict['height'])
                for key in params_dict_6prim.keys():
                    param_to_optimizer_idx_6prim[key] = 0

            print(f"\nStarting optimization with 6 primitives...")
            print(f"Number of parameters: {len(guess_6prim)}")

            # Update prmtop with 6-prim initial guess
            # Convert multi-primitive format to grouped format for ParmEd
            params_dict_6prim_grouped = ConvertMultiPrimParamsForParmEd(params_dict_6prim)
            prmtop_6prim_initial = UpdateParmTopInMemory(prmtop_dir, params_dict_6prim_grouped)
            prmtop_6prim_initial.save(prmtop_dir, overwrite=True)

            # Re-run optimization with 6 primitives
            minimization_result_6prim = minimize(ObjectiveFunction, guess_6prim, jac=True, \
                   args=(coordinates, boxVectors, ref_ene, params_dict_6prim, optvars_dict, prmtopomm, torsion_indices,
                         min_steps, outdir, prmtop_dir, min_interval, crd_flist, geo_dir, amber_dir, loss, loss_delta, param_to_optimizer_idx_6prim), \
                   bounds=bounds_6prim, method=algorithm, options={'maxiter':opt_loops, 'eps': step_size})

            # Compare results
            best_rmsd_6prim = jnp.sqrt(best_loss / len(ref_ene_array))

            print("\n" + "="*60)
            print("FALLBACK RESULTS COMPARISON")
            print("="*60)
            print(f"Best RMSD with 3 primitives: {best_rmsd_3prim:.6f} kcal/mol")
            print(f"Best RMSD with 6 primitives: {best_rmsd_6prim:.6f} kcal/mol")

            if best_rmsd_6prim < best_rmsd_3prim:
                improvement = ((best_rmsd_3prim - best_rmsd_6prim) / best_rmsd_3prim) * 100
                print(f"\n✓ 6-primitive fit IMPROVED by {improvement:.1f}%")
                print(f"  Keeping 6-primitive result (nprim=6)")
                # Use 6-prim results
                params_dict = params_dict_6prim
                minimization_result = minimization_result_6prim
                nprim = 6
            else:
                print(f"\n✗ 6-primitive fit did NOT improve")
                print(f"  Reverting to 3-primitive result (nprim=3)")
                # Restore 3-prim results
                best_loss = best_loss_3prim
                best_params = best_params_3prim
                best_energy = best_energy_3prim
                best_iteration = best_iteration_3prim
                # params_dict already has 3-prim version
            print("="*60)
        else:
            print(f"\n✓ RMSD within threshold, keeping 3 primitives")
            print("="*60)

    x = minimization_result.x

    # Set final parameters in dictionary and save - ONLY heights
    i=0
    for key, value in params_dict.items():
        value['height']=x[i]
        i+=1

    # SaveJsonData(params_dict, outdir + '/final_params.json')

    # AB: Save best energies to a JSON file. Temporary solution. Move function to jaxextract.py script.  
    final_energy_dict = {"best_energy": best_energy}
    SaveJsonData(final_energy_dict, 'energies.json')

    # AB: Save the output files corresponding to the best iteration back to the original folder and cleanup the dir. 
    dest_dir = os.path.dirname(geo_dir)
    cleanup_and_restore_best(outdir, dest_dir, best_iteration)

    return

def main():
    # create parser for command-line arguments
    parser = argparse.ArgumentParser(description='AMBER Torsion Optimizer',
                                   formatter_class=SmartFormatter)

    parser.add_argument('--prmtop', metavar='filename',
      type=str,
      default="../Datasets/amber/dh_6-7-9-11/prmtop",
      help='Location of PRMTOP file')
    parser.add_argument('--params', metavar='filename',
      type=str,
      default="../Datasets/amber/dh_6-7-9-11/params.json",
      help='Parameters file')
    parser.add_argument('--geo', metavar='filename',
      type=str,
      default="../Datasets/amber/dh_6-7-9-11/confs_999-999/dh_6-7-9-11/dh_6-7-9-11",
      help='Directory with geometry files')
    parser.add_argument('--amber_dir', metavar='filename',
      type=str,
      default="../Datasets/amber/dh_6-7-9-11/confs_999-999/dh_6-7-9-11",
      help='Directory with AMBER files')
    parser.add_argument('--reference', metavar='filename',
      type=str,
      default="../Datasets/amber/dh_6-7-9-11/ref_ene.json",
      help='Directory to output results')
    parser.add_argument('--out', metavar='filename',
      type=str,
      default="../Datasets/amber/dh_6-7-9-11/jaxout",
      help='Directory to output results')
    parser.add_argument('--minsteps', metavar='steps',
      type=int,
      default=2000,
      help='Maximum number of energy minimization steps')
    parser.add_argument('--maxiter', metavar='iterations',
      type=int,
      default=1000,
      help='Maximum number of optimization iterations')
    parser.add_argument('--mininterval', metavar='iterations',
      type=int,
      default=5,
      help='Number of parameter optimization iterations between geometry optimization')
    parser.add_argument('--nprim', metavar='primitives',
      type=int,
      default=3,
      help='Number of primitives (periodicities): 1=single, 3=multi-primitive [1,2,3], 6=extended [1,2,3,4,5,6]')
    parser.add_argument('--phase_strategy', metavar='strategy',
      type=str,
      default='fixed_zero',
      choices=['fixed_zero', 'alternating'],
      help='Phase strategy for multi-primitive: fixed_zero (all 0°) or alternating (0°/180°)')
    parser.add_argument('--initial_guess', metavar='method',
      type=str,
      default='linear',
      choices=['linear', 'fourier', 'gaff'],
      help='Initial guess method: linear (LSQ/FFPOpt), fourier (Fourier analysis), gaff (standard GAFF)')
    parser.add_argument('--auto_fallback', metavar='bool',
      type=lambda x: x.lower() in ['true', '1', 'yes'],
      default=True,
      help='Enable automatic fallback to nprim=6 if nprim=3 fails (default: True)')
    parser.add_argument('--fallback_threshold', metavar='threshold',
      type=float,
      default=0.5,
      help='RMSD threshold (kcal/mol) for triggering fallback to nprim=6 (default: 0.5)')
    parser.add_argument('--loss', metavar='function',
      type=str,
      default='huber',
      choices=['linear', 'huber', 'soft_l1', 'cauchy', 'arctan'],
      help='Loss function: linear (SSE), huber (robust), soft_l1, cauchy, arctan (default: huber)')
    parser.add_argument('--loss_delta', metavar='delta',
      type=float,
      default=1.0,
      help='Scaling parameter for robust loss functions (default: 1.0)')
    parser.add_argument('--coupling', metavar='mode',
      type=str,
      default='semi-independent',
      choices=['fully-independent', 'semi-independent', 'fully-coupled'],
      help='Torsion coupling mode: semi-independent (RECOMMENDED), fully-independent, fully-coupled (default: semi-independent)')

    args = parser.parse_args()

    ff_opt(args.prmtop, args.params, args.geo, args.amber_dir, args.minsteps, args.maxiter, args.reference, args.out, args.mininterval, args.nprim, args.phase_strategy, args.initial_guess, args.auto_fallback, args.fallback_threshold, args.loss, args.loss_delta, args.coupling)

if __name__ == "__main__":
    main()

# needs a file with the reference energies as a dictionary, very simple format
# the torsions are read from the params file and then matched with the parameter indices from the prmtop
# it's assumed that the torsion being constrained is the first torsion in this list but i could change this if some other behavior is desired

# have to figure out how to do parmed mods for torsion prms, if 2 torsions have the same params, they end up mapping to the same index, even when parmed updates them
# this isn't good because we want different indices for every torsion, not sure if there's an easy way to force seperate parameter indices to be generated
# the code as is doesn't touch parmed except for the final parameter update so it's assumed that the seperate torsion parameter indices exist before running the optimizer


# if any structures don't display good results, we can look into changing minimization interval, optimizer tolerance, and a few other
# things. there was also the discussion about offloading the constrained minimization to a package with better tools for it and just
# doing the final gradient evaluation in jax at a potential speed hit. constraint parameters can also be tuned with current reax style approach

# could look into doing meta optimization of these penalty term parameters assuming there's good energy or other physical references
# from a better approach to avoid overfitting the restraint by treating loss as angular deviation alone
