#------------------------------------------------------------------------------#
#  fortnet-python: Python Tools for the Fortnet Software Package               #
#  Copyright (C) 2021 - 2022 T. W. van der Heide                               #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
#------------------------------------------------------------------------------#


'''
Implements common functionalities for the fortnet-python regression testsuite.
'''


import warnings
import subprocess
import numpy as np
from ase import Atoms

from fortformat import Fnetout


ATOL = 1e-16
RTOL = 1e-14


class Hdf5:
    '''Representation of an HDF5 file to process.'''


    def __init__(self, fname):
        '''Initializes an Hdf5 file object.

        Args:

            fname (str): path to the HDF5 file to process

        '''

        self._fname = fname


    def equals(self, fname):
        '''Checks equality with another reference instance

        Args:

            fname (str): path to HDF5 file to compare with

        Returns:

            equal (bool): True, if the two instances are equal

        '''

        process = subprocess.run(['h5diff', self._fname, fname], check=False)

        equal = process.returncode == 0

        return equal


def compare_fnetout_references(ref, fname, atol=ATOL, rtol=RTOL):
    '''Compares the properties extracted by using the
       Fnetout class with raw reference values.

    Args:

        ref (dict): expected content of the fnetout file
        fname (str): path to HDF5 file to load and compare with
        atol (float): required absolute tolerance
        rtol (float): required relative tolerance

    Returns:

        equal (bool): true, if extracted properties match references

    '''

    fnetout = Fnetout(fname)

    mode = fnetout.mode
    ndatapoints = fnetout.ndatapoints
    nglobaltargets = fnetout.nglobaltargets
    natomictargets = fnetout.natomictargets
    globaltargets = fnetout.globaltargets
    atomictargets = fnetout.atomictargets
    tforces = fnetout.tforces
    forces = fnetout.forces
    atomicpredictions = fnetout.atomicpredictions
    globalpredictions_atomic = fnetout.globalpredictions_atomic

    equal = mode == ref['mode']

    if not equal:
        warnings.warn('Mismatch in running mode.')
        return False

    equal = ndatapoints == ref['ndatapoints']

    if not equal:
        warnings.warn('Mismatch in number of training datapoints.')
        return False

    equal = nglobaltargets == ref['nglobaltargets']

    if not equal:
        warnings.warn('Mismatch in number of system-wide targets.')
        return False

    equal = natomictargets == ref['natomictargets']

    if not equal:
        warnings.warn('Mismatch in number of atomic targets.')
        return False

    if ref['globaltargets'] is not None:
        for ii, target in enumerate(globaltargets):
            equal = np.allclose(target, ref['globaltargets'][ii],
                                rtol=rtol, atol=atol)

            if not equal:
                warnings.warn('Mismatch in global targets of datapoint ' \
                              + str(ii + 1) + '.')
                return False

    if ref['atomictargets'] is not None:
        for ii, target in enumerate(atomictargets):
            equal = np.allclose(target, ref['atomictargets'][ii],
                                rtol=rtol, atol=atol)

            if not equal:
                warnings.warn('Mismatch in atomic targets of datapoint ' \
                              + str(ii + 1) + '.')
                return False

    equal = tforces == ref['tforces']

    if not equal:
        warnings.warn('Mismatch in force specification.')
        return False

    if ref['forces'] is not None:
        for idata in range(ndatapoints):
            for itarget, force in enumerate(forces[idata]):
                equal = np.allclose(force,
                                    ref['forces'][idata][itarget],
                                    rtol=rtol, atol=atol)

                if not equal:
                    warnings.warn('Mismatch in forces of datapoint ' \
                                  + str(idata + 1) + ' and target ' + \
                                  str(itarget + 1) + '.')
                    return False

    if ref['atomicpredictions'] is not None:
        for ii, prediction in enumerate(atomicpredictions):
            equal = np.allclose(prediction, ref['atomicpredictions'][ii],
                                rtol=rtol, atol=atol)

            if not equal:
                warnings.warn('Mismatch in atomic predictions of datapoint ' \
                              + str(ii + 1) + '.')
                return False

    if ref['globalpredictions_atomic'] is not None:
        for ii, target in enumerate(globalpredictions_atomic):
            equal = np.allclose(target, ref['globalpredictions_atomic'][ii],
                                rtol=rtol, atol=atol)

            if not equal:
                warnings.warn('Mismatch in (atom-resolved) global predictions' \
                              + ' of datapoint ' + str(ii + 1) + '.')
                return False

    return True


def get_mixed_geometries():
    '''Generates six geometries with(out) periodic boundary conditions.'''

    atoms = []
    atoms += get_cluster_geometries()
    atoms += get_bulk_geometries()

    return atoms


def get_cluster_geometries():
    '''Generates three molecules without periodic boundary conditions.'''

    h2o = Atoms('H2O')
    h2o.positions = np.array([[0.0, 0.0, 0.119262], [0.0, 0.763239, -0.477047],
                              [0.0, -0.763239, -0.477047]], dtype=float)
    ch4 = Atoms('CH4')
    ch4.positions = np.array([[0.0, 0.0, 0.0], [0.629118, 0.629118, 0.629118],
                              [-0.629118, -0.629118, 0.629118],
                              [0.629118, -0.629118, -0.629118],
                              [-0.629118, 0.629118, -0.629118]], dtype=float)
    nh3 = Atoms('NH3')
    nh3.positions = np.array([[0.0, 0.0, 0.116489], [0.0, 0.939731, -0.271808],
                              [0.813831, -0.469865, -0.271808],
                              [-0.813831, -0.469865, -0.271808]], dtype=float)

    atoms = [h2o, ch4, nh3]

    return atoms


def get_bulk_geometries():
    '''Generates three crystals with periodic boundary conditions.'''

    si = Atoms('Si64')
    si.set_scaled_positions(np.array([
        [0.8750000000E+00, 0.1250000000E+00, 0.1250000000E+00],
        [0.8750000000E+00, 0.3750000000E+00, 0.3750000000E+00],
        [0.8750000000E+00, 0.6250000000E+00, 0.6250000000E+00],
        [0.8750000000E+00, 0.8750000000E+00, 0.8750000000E+00],
        [0.6250000000E+00, 0.1250000000E+00, 0.3750000000E+00],
        [0.6250000000E+00, 0.3750000000E+00, 0.6250000000E+00],
        [0.6250000000E+00, 0.6250000000E+00, 0.8750000000E+00],
        [0.6250000000E+00, 0.8750000000E+00, 0.1250000000E+00],
        [0.3750000000E+00, 0.1250000000E+00, 0.6250000000E+00],
        [0.3750000000E+00, 0.3750000000E+00, 0.8750000000E+00],
        [0.3750000000E+00, 0.6250000000E+00, 0.1250000000E+00],
        [0.3750000000E+00, 0.8750000000E+00, 0.3750000000E+00],
        [0.1250000000E+00, 0.1250000000E+00, 0.8750000000E+00],
        [0.1250000000E+00, 0.3750000000E+00, 0.1250000000E+00],
        [0.1250000000E+00, 0.6250000000E+00, 0.3750000000E+00],
        [0.1250000000E+00, 0.8750000000E+00, 0.6250000000E+00],
        [0.6250000000E+00, 0.3750000000E+00, 0.1250000000E+00],
        [0.6250000000E+00, 0.6250000000E+00, 0.3750000000E+00],
        [0.6250000000E+00, 0.8750000000E+00, 0.6250000000E+00],
        [0.6250000000E+00, 0.1250000000E+00, 0.8750000000E+00],
        [0.3750000000E+00, 0.3750000000E+00, 0.3750000000E+00],
        [0.3750000000E+00, 0.6250000000E+00, 0.6250000000E+00],
        [0.3750000000E+00, 0.8750000000E+00, 0.8750000000E+00],
        [0.3750000000E+00, 0.1250000000E+00, 0.1250000000E+00],
        [0.1250000000E+00, 0.3750000000E+00, 0.6250000000E+00],
        [0.1250000000E+00, 0.6250000000E+00, 0.8750000000E+00],
        [0.1250000000E+00, 0.8750000000E+00, 0.1250000000E+00],
        [0.1250000000E+00, 0.1250000000E+00, 0.3750000000E+00],
        [0.8750000000E+00, 0.3750000000E+00, 0.8750000000E+00],
        [0.8750000000E+00, 0.6250000000E+00, 0.1250000000E+00],
        [0.8750000000E+00, 0.8750000000E+00, 0.3750000000E+00],
        [0.8750000000E+00, 0.1250000000E+00, 0.6250000000E+00],
        [0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00],
        [0.0000000000E+00, 0.2500000000E+00, 0.2500000000E+00],
        [0.0000000000E+00, 0.5000000000E+00, 0.5000000000E+00],
        [0.0000000000E+00, 0.7500000000E+00, 0.7500000000E+00],
        [0.7500000000E+00, 0.0000000000E+00, 0.2500000000E+00],
        [0.7500000000E+00, 0.2500000000E+00, 0.5000000000E+00],
        [0.7500000000E+00, 0.5000000000E+00, 0.7500000000E+00],
        [0.7500000000E+00, 0.7500000000E+00, 0.0000000000E+00],
        [0.5000000000E+00, 0.0000000000E+00, 0.5000000000E+00],
        [0.5000000000E+00, 0.2500000000E+00, 0.7500000000E+00],
        [0.5000000000E+00, 0.5000000000E+00, 0.0000000000E+00],
        [0.5000000000E+00, 0.7500000000E+00, 0.2500000000E+00],
        [0.2500000000E+00, 0.0000000000E+00, 0.7500000000E+00],
        [0.2500000000E+00, 0.2500000000E+00, 0.0000000000E+00],
        [0.2500000000E+00, 0.5000000000E+00, 0.2500000000E+00],
        [0.2500000000E+00, 0.7500000000E+00, 0.5000000000E+00],
        [0.7500000000E+00, 0.2500000000E+00, 0.0000000000E+00],
        [0.7500000000E+00, 0.5000000000E+00, 0.2500000000E+00],
        [0.7500000000E+00, 0.7500000000E+00, 0.5000000000E+00],
        [0.7500000000E+00, 0.0000000000E+00, 0.7500000000E+00],
        [0.5000000000E+00, 0.2500000000E+00, 0.2500000000E+00],
        [0.5000000000E+00, 0.5000000000E+00, 0.5000000000E+00],
        [0.5000000000E+00, 0.7500000000E+00, 0.7500000000E+00],
        [0.5000000000E+00, 0.0000000000E+00, 0.0000000000E+00],
        [0.2500000000E+00, 0.2500000000E+00, 0.5000000000E+00],
        [0.2500000000E+00, 0.5000000000E+00, 0.7500000000E+00],
        [0.2500000000E+00, 0.7500000000E+00, 0.0000000000E+00],
        [0.2500000000E+00, 0.0000000000E+00, 0.2500000000E+00],
        [0.0000000000E+00, 0.2500000000E+00, 0.7500000000E+00],
        [0.0000000000E+00, 0.5000000000E+00, 0.0000000000E+00],
        [0.0000000000E+00, 0.7500000000E+00, 0.2500000000E+00],
        [0.0000000000E+00, 0.0000000000E+00, 0.5000000000E+00]], dtype=float))
    si.set_cell(np.array([
        [0.8764470533E+01, 0.0000000000E+00, 0.0000000000E+00],
        [0.0000000000E+00, 0.8764470533E+01, 0.0000000000E+00],
        [0.0000000000E+00, 0.0000000000E+00, 0.8764470533E+01]], dtype=float))

    sic = Atoms('Si32C32')
    sic.set_scaled_positions(np.array([
        [0.8750000000E+00, 0.1250000000E+00, 0.1250000000E+00],
        [0.8750000000E+00, 0.3750000000E+00, 0.3750000000E+00],
        [0.8750000000E+00, 0.6250000000E+00, 0.6250000000E+00],
        [0.8750000000E+00, 0.8750000000E+00, 0.8750000000E+00],
        [0.6250000000E+00, 0.1250000000E+00, 0.3750000000E+00],
        [0.6250000000E+00, 0.3750000000E+00, 0.6250000000E+00],
        [0.6250000000E+00, 0.6250000000E+00, 0.8750000000E+00],
        [0.6250000000E+00, 0.8750000000E+00, 0.1250000000E+00],
        [0.3750000000E+00, 0.1250000000E+00, 0.6250000000E+00],
        [0.3750000000E+00, 0.3750000000E+00, 0.8750000000E+00],
        [0.3750000000E+00, 0.6250000000E+00, 0.1250000000E+00],
        [0.3750000000E+00, 0.8750000000E+00, 0.3750000000E+00],
        [0.1250000000E+00, 0.1250000000E+00, 0.8750000000E+00],
        [0.1250000000E+00, 0.3750000000E+00, 0.1250000000E+00],
        [0.1250000000E+00, 0.6250000000E+00, 0.3750000000E+00],
        [0.1250000000E+00, 0.8750000000E+00, 0.6250000000E+00],
        [0.6250000000E+00, 0.3750000000E+00, 0.1250000000E+00],
        [0.6250000000E+00, 0.6250000000E+00, 0.3750000000E+00],
        [0.6250000000E+00, 0.8750000000E+00, 0.6250000000E+00],
        [0.6250000000E+00, 0.1250000000E+00, 0.8750000000E+00],
        [0.3750000000E+00, 0.3750000000E+00, 0.3750000000E+00],
        [0.3750000000E+00, 0.6250000000E+00, 0.6250000000E+00],
        [0.3750000000E+00, 0.8750000000E+00, 0.8750000000E+00],
        [0.3750000000E+00, 0.1250000000E+00, 0.1250000000E+00],
        [0.1250000000E+00, 0.3750000000E+00, 0.6250000000E+00],
        [0.1250000000E+00, 0.6250000000E+00, 0.8750000000E+00],
        [0.1250000000E+00, 0.8750000000E+00, 0.1250000000E+00],
        [0.1250000000E+00, 0.1250000000E+00, 0.3750000000E+00],
        [0.8750000000E+00, 0.3750000000E+00, 0.8750000000E+00],
        [0.8750000000E+00, 0.6250000000E+00, 0.1250000000E+00],
        [0.8750000000E+00, 0.8750000000E+00, 0.3750000000E+00],
        [0.8750000000E+00, 0.1250000000E+00, 0.6250000000E+00],
        [0.0000000000E+00, 0.0000000000E+00, 0.0000000000E+00],
        [0.0000000000E+00, 0.2500000000E+00, 0.2500000000E+00],
        [0.0000000000E+00, 0.5000000000E+00, 0.5000000000E+00],
        [0.0000000000E+00, 0.7500000000E+00, 0.7500000000E+00],
        [0.7500000000E+00, 0.0000000000E+00, 0.2500000000E+00],
        [0.7500000000E+00, 0.2500000000E+00, 0.5000000000E+00],
        [0.7500000000E+00, 0.5000000000E+00, 0.7500000000E+00],
        [0.7500000000E+00, 0.7500000000E+00, 0.0000000000E+00],
        [0.5000000000E+00, 0.0000000000E+00, 0.5000000000E+00],
        [0.5000000000E+00, 0.2500000000E+00, 0.7500000000E+00],
        [0.5000000000E+00, 0.5000000000E+00, 0.0000000000E+00],
        [0.5000000000E+00, 0.7500000000E+00, 0.2500000000E+00],
        [0.2500000000E+00, 0.0000000000E+00, 0.7500000000E+00],
        [0.2500000000E+00, 0.2500000000E+00, 0.0000000000E+00],
        [0.2500000000E+00, 0.5000000000E+00, 0.2500000000E+00],
        [0.2500000000E+00, 0.7500000000E+00, 0.5000000000E+00],
        [0.7500000000E+00, 0.2500000000E+00, 0.0000000000E+00],
        [0.7500000000E+00, 0.5000000000E+00, 0.2500000000E+00],
        [0.7500000000E+00, 0.7500000000E+00, 0.5000000000E+00],
        [0.7500000000E+00, 0.0000000000E+00, 0.7500000000E+00],
        [0.5000000000E+00, 0.2500000000E+00, 0.2500000000E+00],
        [0.5000000000E+00, 0.5000000000E+00, 0.5000000000E+00],
        [0.5000000000E+00, 0.7500000000E+00, 0.7500000000E+00],
        [0.5000000000E+00, 0.0000000000E+00, 0.0000000000E+00],
        [0.2500000000E+00, 0.2500000000E+00, 0.5000000000E+00],
        [0.2500000000E+00, 0.5000000000E+00, 0.7500000000E+00],
        [0.2500000000E+00, 0.7500000000E+00, 0.0000000000E+00],
        [0.2500000000E+00, 0.0000000000E+00, 0.2500000000E+00],
        [0.0000000000E+00, 0.2500000000E+00, 0.7500000000E+00],
        [0.0000000000E+00, 0.5000000000E+00, 0.0000000000E+00],
        [0.0000000000E+00, 0.7500000000E+00, 0.2500000000E+00],
        [0.0000000000E+00, 0.0000000000E+00, 0.5000000000E+00]], dtype=float))
    sic.set_cell(np.array([
        [0.8764470533E+01, 0.0000000000E+00, 0.0000000000E+00],
        [0.0000000000E+00, 0.8764470533E+01, 0.0000000000E+00],
        [0.0000000000E+00, 0.0000000000E+00, 0.8764470533E+01]], dtype=float))

    cu = Atoms('Cu')
    cu.set_positions(np.array([[0.0, 0.0, 0.0]], dtype=float))
    cu.set_cell(np.array([[0.0, 1.8, 1.8], [1.8, 0.0, 1.8], [1.8, 1.8, 0.0]],
                         dtype=float))

    atoms = [si, sic, cu]

    return atoms


def get_properties_byatoms(atoms, nprops, atomic):
    '''Generates dummy properties for regression testing.

    Args:

        atoms (ASE atoms list): list of ASE Atoms objects
        nprops (int): number of atomic or system-wide properties
        atomic (bool): true, if atomic properties are desired

    Returns:

        props (list or 2darray): atomic or system wide-properties

    '''

    # fix random seed for reproduction purposes
    np.random.seed(42)

    if atomic:
        props = []
        for atom in atoms:
            natom = len(atom)
            props.append(np.random.random((natom, nprops)))
    else:
        props = np.random.random((len(atoms), nprops))

    return props


def get_properties_byfeatures(features, nprops, atomic):
    '''Generates dummy properties for regression testing.

    Args:

        features (list): list of external atomic input features
        nprops (int): number of atomic or system-wide properties
        atomic (bool): true, if atomic properties are desired

    '''

    # fix random seed for reproduction purposes
    np.random.seed(42)

    if atomic:
        props = []
        for feature in features:
            natom = np.shape(feature)[1]
            props.append(np.random.random((natom, nprops)))
    else:
        props = np.random.random((len(features), nprops))

    return props


def get_atomicweights_byatoms(atoms):
    '''Generates dummy properties for regression testing.

    Args:

        atoms (ASE atoms list): list of ASE Atoms objects

    Returns:

        weights (list): atomic gradient weighting

    '''

    # fix random seed for reproduction purposes
    np.random.seed(42)

    weights = []
    for atom in atoms:
        natom = len(atom)
        weights.append(np.asfarray(np.random.randint(1, 100, natom, dtype=int)))

    return weights


def get_batomicweights_byatoms(atoms):
    '''Generates dummy properties for regression testing.

    Args:

        atoms (ASE atoms list): list of ASE Atoms objects

    Returns:

        weights (list): atomic gradient weighting

    '''

    # fix random seed for reproduction purposes
    np.random.seed(42)
    sample = [True, False]

    weights = []
    for atom in atoms:
        natom = len(atom)
        weights.append(np.random.choice(sample, size=natom))

    return weights
