#!/usr/bin/env python3
#------------------------------------------------------------------------------#
#  fortnet-python: Python Tools for the Fortnet Software Package               #
#  Copyright (C) 2021 - 2025 T. W. van der Heide                               #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
#------------------------------------------------------------------------------#


'''
Regression tests covering the Fnetdata class of Fortformat.
'''


import os
import pytest
import numpy as np
from fortformat import Fnetdata

from common import Hdf5, get_cluster_geometries, get_bulk_geometries, \
    get_mixed_geometries, get_properties_byatoms, get_atomicweights_byatoms, \
    get_batomicweights_byatoms


REFPATH = os.path.join(os.getcwd(), 'test', 'references', 'Fnetdata')


def test_csgeometries(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: No
        atomic targets: No
        external features: No
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'csgeometries.hdf5'
    csatoms = get_mixed_geometries()

    fnetdata = Fnetdata(atoms=csatoms)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: No
        atomic targets: No
        external features: No
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'cgeometries.hdf5'
    catoms = get_cluster_geometries()

    fnetdata = Fnetdata(atoms=catoms)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_extfeatures(tmpdir):
    '''Test dataset generation for configuration:

        structures: No
        periodic: /
        global targets: No
        atomic targets: No
        external features: Yes
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'extfeatures.hdf5'
    features = []

    # fix random seed for reproduction purposes
    np.random.seed(42)

    for natom in [3, 7, 13]:
        features.append(np.random.random((natom, 4)))

    fnetdata = Fnetdata(features=features)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries_extfeatures(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: No
        atomic targets: No
        external features: Yes
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'cgeometries_extfeatures.hdf5'
    catoms = get_cluster_geometries()
    features = get_properties_byatoms(catoms, 5, True)

    fnetdata = Fnetdata(atoms=catoms, features=features)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries_weights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: No
        atomic targets: No
        external features: No
        manual dataset weights: Yes
        manual gradient weights: No

    '''

    fname = 'cgeometries_weights.hdf5'
    catoms = get_cluster_geometries()
    # fix random seed for reproduction purposes
    np.random.seed(42)
    weights = np.random.randint(1, 100, len(catoms), dtype=int)

    fnetdata = Fnetdata(atoms=catoms)
    fnetdata.weights = weights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries_atomictargets(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: No
        atomic targets: Yes
        external features: No
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'cgeometries_atomictargets.hdf5'
    catoms = get_cluster_geometries()
    targets = get_properties_byatoms(catoms, 3, True)

    fnetdata = Fnetdata(atoms=catoms, atomictargets=targets)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries_atomictargets_extfeatures(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: No
        atomic targets: Yes
        external features: Yes
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'cgeometries_atomictargets_extfeatures.hdf5'
    catoms = get_cluster_geometries()
    targets = get_properties_byatoms(catoms, 3, True)
    features = get_properties_byatoms(catoms, 13, True)

    fnetdata = Fnetdata(atoms=catoms, features=features, atomictargets=targets)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries_atomictargets_weights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: No
        atomic targets: Yes
        external features: No
        manual dataset weights: Yes
        manual gradient weights: No

    '''

    fname = 'cgeometries_atomictargets_weights.hdf5'
    catoms = get_cluster_geometries()
    # fix random seed for reproduction purposes
    np.random.seed(42)
    weights = np.random.randint(1, 100, len(catoms), dtype=int)
    targets = get_properties_byatoms(catoms, 3, True)

    fnetdata = Fnetdata(atoms=catoms, atomictargets=targets)
    fnetdata.weights = weights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries_atomictargets_atomicweights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: No
        atomic targets: Yes
        external features: No
        manual dataset weights: No
        manual gradient weights: Yes

    '''

    fname = 'cgeometries_atomictargets_atomicweights.hdf5'
    catoms = get_cluster_geometries()
    atomicweights = get_atomicweights_byatoms(catoms)
    targets = get_properties_byatoms(catoms, 3, True)

    fnetdata = Fnetdata(atoms=catoms, atomictargets=targets)
    fnetdata.atomicweights = atomicweights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries_atomictargets_batomicweights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: No
        atomic targets: Yes
        external features: No
        manual dataset weights: No
        manual gradient weights: Yes

    '''

    fname = 'cgeometries_atomictargets_batomicweights.hdf5'
    catoms = get_cluster_geometries()
    batomicweights = get_batomicweights_byatoms(catoms)
    targets = get_properties_byatoms(catoms, 3, True)

    fnetdata = Fnetdata(atoms=catoms, atomictargets=targets)
    fnetdata.atomicweights = batomicweights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries_globaltargets(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: Yes
        atomic targets: No
        external features: No
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'cgeometries_globaltargets.hdf5'
    catoms = get_cluster_geometries()
    targets = get_properties_byatoms(catoms, 3, False)

    fnetdata = Fnetdata(atoms=catoms, globaltargets=targets)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries_globaltargets_extfeatures(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: Yes
        atomic targets: No
        external features: Yes
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'cgeometries_globaltargets_extfeatures.hdf5'
    catoms = get_cluster_geometries()
    targets = get_properties_byatoms(catoms, 3, False)
    features = get_properties_byatoms(catoms, 13, True)

    fnetdata = Fnetdata(atoms=catoms, features=features, globaltargets=targets)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries_globaltargets_weights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: Yes
        atomic targets: No
        external features: No
        manual dataset weights: Yes
        manual gradient weights: No

    '''

    fname = 'cgeometries_globaltargets_weights.hdf5'
    catoms = get_cluster_geometries()
    # fix random seed for reproduction purposes
    np.random.seed(42)
    weights = np.random.randint(1, 100, len(catoms), dtype=int)
    targets = get_properties_byatoms(catoms, 3, False)

    fnetdata = Fnetdata(atoms=catoms, globaltargets=targets)
    fnetdata.weights = weights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries_globaltargets_atomicweights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: Yes
        atomic targets: No
        external features: No
        manual dataset weights: No
        manual gradient weights: Yes

    '''

    fname = 'cgeometries_globaltargets_atomicweights.hdf5'
    catoms = get_cluster_geometries()
    atomicweights = get_atomicweights_byatoms(catoms)
    targets = get_properties_byatoms(catoms, 3, False)

    fnetdata = Fnetdata(atoms=catoms, globaltargets=targets)
    fnetdata.atomicweights = atomicweights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries_globaltargets_batomicweights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: Yes
        atomic targets: No
        external features: No
        manual dataset weights: No
        manual gradient weights: Yes

    '''

    fname = 'cgeometries_globaltargets_batomicweights.hdf5'
    catoms = get_cluster_geometries()
    batomicweights = get_batomicweights_byatoms(catoms)
    targets = get_properties_byatoms(catoms, 3, False)

    fnetdata = Fnetdata(atoms=catoms, globaltargets=targets)
    fnetdata.atomicweights = batomicweights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_sgeometries_extfeatures(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: Yes
        global targets: No
        atomic targets: No
        external features: Yes
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'sgeometries_extfeatures.hdf5'
    satoms = get_bulk_geometries()
    features = get_properties_byatoms(satoms, 16, True)

    fnetdata = Fnetdata(atoms=satoms, features=features)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_sgeometries_atomictargets(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: Yes
        global targets: No
        atomic targets: Yes
        external features: No
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'sgeometries_atomictargets.hdf5'
    satoms = get_bulk_geometries()
    targets = get_properties_byatoms(satoms, 3, True)

    fnetdata = Fnetdata(atoms=satoms, atomictargets=targets)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_sgeometries_atomictargets_extfeatures(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: Yes
        global targets: No
        atomic targets: Yes
        external features: Yes
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'sgeometries_atomictargets_extfeatures.hdf5'
    satoms = get_bulk_geometries()
    targets = get_properties_byatoms(satoms, 3, True)
    features = get_properties_byatoms(satoms, 13, True)

    fnetdata = Fnetdata(atoms=satoms, features=features, atomictargets=targets)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_sgeometries_atomictargets_weights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: Yes
        global targets: No
        atomic targets: Yes
        external features: No
        manual dataset weights: Yes
        manual gradient weights: No

    '''

    fname = 'sgeometries_atomictargets_weights.hdf5'
    satoms = get_bulk_geometries()
    # fix random seed for reproduction purposes
    np.random.seed(42)
    weights = np.random.randint(1, 100, len(satoms), dtype=int)
    targets = get_properties_byatoms(satoms, 3, True)

    fnetdata = Fnetdata(atoms=satoms, atomictargets=targets)
    fnetdata.weights = weights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_sgeometries_atomictargets_atomicweights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: Yes
        global targets: No
        atomic targets: Yes
        external features: No
        manual dataset weights: No
        manual gradient weights: Yes

    '''

    fname = 'sgeometries_atomictargets_atomicweights.hdf5'
    satoms = get_bulk_geometries()
    atomicweights = get_atomicweights_byatoms(satoms)
    targets = get_properties_byatoms(satoms, 3, True)

    fnetdata = Fnetdata(atoms=satoms, atomictargets=targets)
    fnetdata.atomicweights = atomicweights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_sgeometries_atomictargets_batomicweights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: Yes
        global targets: No
        atomic targets: Yes
        external features: No
        manual dataset weights: No
        manual gradient weights: Yes

    '''

    fname = 'sgeometries_atomictargets_batomicweights.hdf5'
    satoms = get_bulk_geometries()
    batomicweights = get_batomicweights_byatoms(satoms)
    targets = get_properties_byatoms(satoms, 3, True)

    fnetdata = Fnetdata(atoms=satoms, atomictargets=targets)
    fnetdata.atomicweights = batomicweights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_sgeometries_globaltargets(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: Yes
        global targets: Yes
        atomic targets: No
        external features: No
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'sgeometries_globaltargets.hdf5'
    satoms = get_bulk_geometries()
    targets = get_properties_byatoms(satoms, 3, False)

    fnetdata = Fnetdata(atoms=satoms, globaltargets=targets)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_sgeometries_globaltargets_extfeatures(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: Yes
        global targets: Yes
        atomic targets: No
        external features: Yes
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'sgeometries_globaltargets_extfeatures.hdf5'
    satoms = get_bulk_geometries()
    targets = get_properties_byatoms(satoms, 3, False)
    features = get_properties_byatoms(satoms, 13, True)

    fnetdata = Fnetdata(atoms=satoms, features=features, globaltargets=targets)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_sgeometries_globaltargets_weights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: Yes
        global targets: Yes
        atomic targets: No
        external features: No
        manual dataset weights: Yes
        manual gradient weights: No

    '''

    fname = 'sgeometries_globaltargets_weights.hdf5'
    satoms = get_bulk_geometries()
    # fix random seed for reproduction purposes
    np.random.seed(42)
    weights = np.random.randint(1, 100, len(satoms), dtype=int)
    targets = get_properties_byatoms(satoms, 3, False)

    fnetdata = Fnetdata(atoms=satoms, globaltargets=targets)
    fnetdata.weights = weights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_sgeometries_globaltargets_atomicweights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: Yes
        global targets: Yes
        atomic targets: No
        external features: No
        manual dataset weights: No
        manual gradient weights: Yes

    '''

    fname = 'sgeometries_globaltargets_atomicweights.hdf5'
    satoms = get_bulk_geometries()
    atomicweights = get_atomicweights_byatoms(satoms)
    targets = get_properties_byatoms(satoms, 3, False)

    fnetdata = Fnetdata(atoms=satoms, globaltargets=targets)
    fnetdata.atomicweights = atomicweights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_sgeometries_globaltargets_batomicweights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: Yes
        global targets: Yes
        atomic targets: No
        external features: No
        manual dataset weights: No
        manual gradient weights: Yes

    '''

    fname = 'sgeometries_globaltargets_batomicweights.hdf5'
    satoms = get_bulk_geometries()
    batomicweights = get_batomicweights_byatoms(satoms)
    targets = get_properties_byatoms(satoms, 3, False)

    fnetdata = Fnetdata(atoms=satoms, globaltargets=targets)
    fnetdata.atomicweights = batomicweights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


#######################################################################


def test_cgeometries_globaltargets_atomictargets(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: Yes
        atomic targets: Yes
        external features: No
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'cgeometries_globaltargets_atomictargets.hdf5'
    catoms = get_cluster_geometries()
    globaltargets = get_properties_byatoms(catoms, 3, False)
    atomictargets = get_properties_byatoms(catoms, 3, True)

    fnetdata = Fnetdata(atoms=catoms, globaltargets=globaltargets,
                        atomictargets=atomictargets)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries_globaltargets_atomictargets_extfeatures(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: Yes
        atomic targets: Yes
        external features: Yes
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'cgeometries_globaltargets_atomictargets_extfeatures.hdf5'
    catoms = get_cluster_geometries()
    globaltargets = get_properties_byatoms(catoms, 3, False)
    atomictargets = get_properties_byatoms(catoms, 3, True)
    features = get_properties_byatoms(catoms, 13, True)

    fnetdata = Fnetdata(atoms=catoms, features=features,
                        globaltargets=globaltargets,
                        atomictargets=atomictargets)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries_globaltargets_atomictargets_weights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: Yes
        atomic targets: Yes
        external features: No
        manual dataset weights: Yes
        manual gradient weights: No

    '''

    fname = 'cgeometries_globaltargets_atomictargets_weights.hdf5'
    catoms = get_cluster_geometries()
    # fix random seed for reproduction purposes
    np.random.seed(42)
    weights = np.random.randint(1, 100, len(catoms), dtype=int)
    globaltargets = get_properties_byatoms(catoms, 3, False)
    atomictargets = get_properties_byatoms(catoms, 3, True)

    fnetdata = Fnetdata(atoms=catoms, globaltargets=globaltargets,
                        atomictargets=atomictargets)
    fnetdata.weights = weights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries_globaltargets_atomictargets_atomicweights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: Yes
        atomic targets: Yes
        external features: No
        manual dataset weights: No
        manual gradient weights: Yes

    '''

    fname = 'cgeometries_globaltargets_atomictargets_atomicweights.hdf5'
    catoms = get_cluster_geometries()
    atomicweights = get_atomicweights_byatoms(catoms)
    globaltargets = get_properties_byatoms(catoms, 3, False)
    atomictargets = get_properties_byatoms(catoms, 3, True)

    fnetdata = Fnetdata(atoms=catoms, globaltargets=globaltargets,
                        atomictargets=atomictargets)
    fnetdata.atomicweights = atomicweights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_cgeometries_globaltargets_atomictargets_batomicweights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: No
        global targets: Yes
        atomic targets: Yes
        external features: No
        manual dataset weights: No
        manual gradient weights: Yes

    '''

    fname = 'cgeometries_globaltargets_atomictargets_batomicweights.hdf5'
    catoms = get_cluster_geometries()
    batomicweights = get_batomicweights_byatoms(catoms)
    globaltargets = get_properties_byatoms(catoms, 3, False)
    atomictargets = get_properties_byatoms(catoms, 3, True)

    fnetdata = Fnetdata(atoms=catoms, globaltargets=globaltargets,
                        atomictargets=atomictargets)
    fnetdata.atomicweights = batomicweights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_sgeometries_globaltargets_atomictargets(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: Yes
        global targets: Yes
        atomic targets: Yes
        external features: No
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'sgeometries_globaltargets_atomictargets.hdf5'
    satoms = get_bulk_geometries()
    globaltargets = get_properties_byatoms(satoms, 3, False)
    atomictargets = get_properties_byatoms(satoms, 3, True)

    fnetdata = Fnetdata(atoms=satoms, globaltargets=globaltargets,
                        atomictargets=atomictargets)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_sgeometries_globaltargets_atomictargets_extfeatures(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: Yes
        global targets: Yes
        atomic targets: Yes
        external features: Yes
        manual dataset weights: No
        manual gradient weights: No

    '''

    fname = 'sgeometries_globaltargets_atomictargets_extfeatures.hdf5'
    satoms = get_bulk_geometries()
    globaltargets = get_properties_byatoms(satoms, 3, False)
    atomictargets = get_properties_byatoms(satoms, 3, True)
    features = get_properties_byatoms(satoms, 13, True)

    fnetdata = Fnetdata(atoms=satoms, features=features,
                        globaltargets=globaltargets,
                        atomictargets=atomictargets)
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_sgeometries_globaltargets_atomictargets_weights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: Yes
        global targets: Yes
        atomic targets: Yes
        external features: No
        manual dataset weights: Yes
        manual gradient weights: No

    '''

    fname = 'sgeometries_globaltargets_atomictargets_weights.hdf5'
    satoms = get_bulk_geometries()
    # fix random seed for reproduction purposes
    np.random.seed(42)
    weights = np.random.randint(1, 100, len(satoms), dtype=int)
    globaltargets = get_properties_byatoms(satoms, 3, False)
    atomictargets = get_properties_byatoms(satoms, 3, True)

    fnetdata = Fnetdata(atoms=satoms, globaltargets=globaltargets,
                        atomictargets=atomictargets)
    fnetdata.weights = weights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_sgeometries_globaltargets_atomictargets_atomicweights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: Yes
        global targets: Yes
        atomic targets: Yes
        external features: No
        manual dataset weights: No
        manual gradient weights: Yes

    '''

    fname = 'sgeometries_globaltargets_atomictargets_atomicweights.hdf5'
    satoms = get_bulk_geometries()
    atomicweights = get_atomicweights_byatoms(satoms)
    globaltargets = get_properties_byatoms(satoms, 3, False)
    atomictargets = get_properties_byatoms(satoms, 3, True)

    fnetdata = Fnetdata(atoms=satoms, globaltargets=globaltargets,
                        atomictargets=atomictargets)
    fnetdata.atomicweights = atomicweights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


def test_sgeometries_globaltargets_atomictargets_batomicweights(tmpdir):
    '''Test dataset generation for configuration:

        structures: Yes
        periodic: Yes
        global targets: Yes
        atomic targets: Yes
        external features: No
        manual dataset weights: No
        manual gradient weights: Yes

    '''

    fname = 'sgeometries_globaltargets_atomictargets_batomicweights.hdf5'
    satoms = get_bulk_geometries()
    batomicweights = get_batomicweights_byatoms(satoms)
    globaltargets = get_properties_byatoms(satoms, 3, False)
    atomictargets = get_properties_byatoms(satoms, 3, True)

    fnetdata = Fnetdata(atoms=satoms, globaltargets=globaltargets,
                        atomictargets=atomictargets)
    fnetdata.atomicweights = batomicweights
    fnetdata.dump(os.path.join(tmpdir, fname))

    hdf5 = Hdf5(os.path.join(tmpdir, fname))
    equal = hdf5.equals(os.path.join(REFPATH, '_' + fname))

    assert equal, 'h5diff reports mismatch in generated datasets.'


if __name__ == '__main__':
    pytest.main()
