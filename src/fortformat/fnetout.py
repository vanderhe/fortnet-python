#------------------------------------------------------------------------------#
#  fortnet-python: Python Tools for the Fortnet Software Package               #
#  Copyright (C) 2021 - 2022 T. W. van der Heide                               #
#                                                                              #
#  See the LICENSE file for terms of usage and distribution.                   #
#------------------------------------------------------------------------------#


'''
Basic Fortnet Output Format Class

This basic Python class implements the Fortnet output file format. The Fnetout
class extracts certain properties of the HDF5 output for later analysis.
'''


import h5py
import numpy as np


class Fnetout:
    '''Basic Fortnet Output Format Class.'''


    def __init__(self, fname):
        '''Initializes a Fnetout object.

        Args:

            fname (str): filename to extract data from

        '''

        self._fname = fname

        with h5py.File(self._fname, 'r') as fnetoutfile:
            fnetout = fnetoutfile['fnetout']
            self._mode = fnetout.attrs.get('mode').decode('UTF-8').strip()
            if not self._mode in ('validate', 'predict'):
                raise FnetoutError('Invalid running mode specification.')

            output = fnetoutfile['fnetout']['output']

            # read number of datapoints
            self._ndatapoints = output.attrs.get('ndatapoints')
            if len(self._ndatapoints) == 1:
                # number of datapoints stored in array of size 1
                self._ndatapoints = self._ndatapoints[0]
            else:
                msg = "Error while reading fnetout file '" + self._fname + \
                    "'. Unrecognized number of datapoints obtained."
                raise FnetoutError(msg)

            # read number of system-wide targets
            self._nglobaltargets = output.attrs.get('nglobaltargets')
            if len(self._nglobaltargets) == 1:
                # number of system-wide targets stored in array of size 1
                self._nglobaltargets = self._nglobaltargets[0]
            else:
                msg = "Error while reading fnetout file '" + self._fname + \
                    "'. Unrecognized number of global targets obtained."
                raise FnetoutError(msg)

            # read number of atomic targets
            self._natomictargets = output.attrs.get('natomictargets')
            if len(self._natomictargets) == 1:
                # number of atomic targets stored in array of size 1
                self._natomictargets = self._natomictargets[0]
            else:
                msg = "Error while reading fnetout file '" + self._fname + \
                    "'. Unrecognized number of atomic targets obtained."
                raise FnetoutError(msg)

            # read force specification
            self._tforces = output.attrs.get('tforces')
            # account for legacy files where no force entry is present
            if self._tforces is None:
                self._tforces = [0]
            if len(self._tforces) == 1:
                # booleans stored in integer arrays of size 1
                self._tforces = bool(self._tforces[0])
            else:
                msg = "Error while reading fnetout file '" + self._fname + \
                    "'. Unrecognized force specification obtained."
                raise FnetoutError(msg)


    @property
    def mode(self):
        '''Defines property, providing the mode of the Fortnet run.

        Returns:

            mode (str): mode of the run that produced the Fnetout file

        '''

        return self._mode


    @property
    def ndatapoints(self):
        '''Defines property, providing the number of datapoints.

        Returns:

            ndatapoints (int): total number of datapoints of the training

        '''

        return self._ndatapoints


    @property
    def nglobaltargets(self):
        '''Defines property, providing the number of system-wide targets.

        Returns:

            nglobaltargets (int): number of global targets per datapoint

        '''

        return self._nglobaltargets


    @property
    def natomictargets(self):
        '''Defines property, providing the number of atomic targets.

        Returns:

            natomictargets (int): number of atomic targets per datapoint

        '''

        return self._natomictargets


    @property
    def tforces(self):
        '''Defines property, providing hint whether atomic forces are present.

        Returns:

            tforces (bool): true, if atomic forces are supplied

        '''

        return self._tforces


    @property
    def globalpredictions(self):
        '''Defines property, providing the system-wide predictions of Fortnet.

        Returns:

            predictions (2darray): predictions of the network

        '''

        if not self._nglobaltargets > 0:
            return None

        predictions = np.empty((self._ndatapoints, self._nglobaltargets),
                               dtype=float)

        with h5py.File(self._fname, 'r') as fnetoutfile:
            output = fnetoutfile['fnetout']['output']
            for idata in range(self._ndatapoints):
                dataname = 'datapoint' + str(idata + 1)
                predictions[idata, :] = np.array(
                    output[dataname]['globalpredictions'],
                    dtype=float)

        return predictions


    @property
    def globalpredictions_atomic(self):
        '''Defines property, providing the (atom-resolved) system-wide
           predictions of Fortnet.

        Returns:

            predictions (list): predictions of the network

        '''

        if not self._nglobaltargets > 0:
            return None

        predictions = []

        with h5py.File(self._fname, 'r') as fnetoutfile:
            output = fnetoutfile['fnetout']['output']
            for idata in range(self._ndatapoints):
                dataname = 'datapoint' + str(idata + 1)
                predictions.append(
                    np.array(output[dataname]['rawpredictions'],
                             dtype=float)[:, 0:self._nglobaltargets])

        return predictions


    @property
    def atomicpredictions(self):
        '''Defines property, providing the atomic predictions of Fortnet.

        Returns:

            predictions (list): predictions of the network

        '''

        if not self._natomictargets > 0:
            return None

        predictions = []

        with h5py.File(self._fname, 'r') as fnetoutfile:
            output = fnetoutfile['fnetout']['output']
            for idata in range(self._ndatapoints):
                dataname = 'datapoint' + str(idata + 1)
                predictions.append(
                    np.array(output[dataname]
                             ['rawpredictions'], dtype=float)
                    [:, self._nglobaltargets:])

        return predictions


    @property
    def globaltargets(self):
        '''Defines property, providing the system-wide targets during training.

        Returns:

            targets (2darray): system-wide targets during training

        '''

        if self._mode == 'predict' or self._nglobaltargets == 0:
            return None

        targets = np.empty((self._ndatapoints, self._nglobaltargets),
                           dtype=float)

        with h5py.File(self._fname, 'r') as fnetoutfile:
            output = fnetoutfile['fnetout']['output']
            for idata in range(self._ndatapoints):
                dataname = 'datapoint' + str(idata + 1)
                targets[idata, :] = np.array(
                    output[dataname]['globaltargets'],
                    dtype=float)

        return targets


    @property
    def atomictargets(self):
        '''Defines property, providing the atomic targets during training.

        Returns:

            targets (list): atomic targets during training

        '''

        if self._mode == 'predict' or self._natomictargets == 0:
            return None

        targets = []

        with h5py.File(self._fname, 'r') as fnetoutfile:
            output = fnetoutfile['fnetout']['output']
            for idata in range(self._ndatapoints):
                dataname = 'datapoint' + str(idata + 1)
                targets.append(np.array(output[dataname]
                                        ['atomictargets'], dtype=float))

        return targets


    @property
    def forces(self):
        '''Defines property, providing the atomic forces, if supplied.

        Returns:

            forces (list): atomic forces on atoms

        '''

        if not self._tforces:
            return None

        tmp1 = []

        if self._natomictargets > 0:
            msg = "Error while extracting forces from fnetout file '" \
                + self._fname + \
                "'. Forces supplied for global property targets only."
            raise FnetoutError(msg)

        with h5py.File(self._fname, 'r') as fnetoutfile:
            output = fnetoutfile['fnetout']['output']
            for idata in range(self._ndatapoints):
                dataname = 'datapoint' + str(idata + 1)
                tmp1.append(np.array(output[dataname]['forces'], dtype=float))

        # convert to shape np.shape(forces[iData][iTarget]) = (iAtom, 3)
        forces = []
        for tmp2 in tmp1:
            entry = []
            if not np.shape(tmp2)[1]%3 == 0:
                msg = "Error while extracting forces from fnetout file '" \
                    + self._fname + \
                    "'. Expected three force components and global target."
                raise FnetoutError(msg)
            for jj in range(int(np.shape(tmp2)[1] / 3)):
                entry.append(tmp2[:, 3 * jj:3 * (jj + 1)])
            forces.append(entry)

        return forces


class FnetoutError(Exception):
    '''Exception thrown by the Fnetout class.'''
