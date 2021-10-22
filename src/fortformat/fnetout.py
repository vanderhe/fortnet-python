#------------------------------------------------------------------------------#
#  fortnet-python: Python Tools for the Fortnet Software Package               #
#  Copyright (C) 2021 T. W. van der Heide                                      #
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
            self._ndatapoints = output.attrs.get('ndatapoints')
            if len(self._ndatapoints) == 1:
                # number of datapoints stored in array of size 1
                self._ndatapoints = self._ndatapoints[0]
            else:
                msg = "Error while reading fnetout file '" + self._fname + \
                    "'. Unrecognized number of datapoints obtained."
                raise FnetoutError(msg)
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

            self._targettype = \
                output.attrs.get('targettype').decode('UTF-8').strip()
            if not self._targettype in ('atomic', 'global'):
                raise FnetoutError('Invalid running mode obtained.')

            # get number of atomic or global predictions/targets
            self._npredictions = np.shape(
                np.array(output['datapoint1']['output']))[1]

            if self._mode == 'validate':
                self._npredictions = int(self._npredictions / 2)


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
    def targettype(self):
        '''Defines property, providing the target type.

        Returns:

            targettype (str): type of targets the network was trained on

        '''

        return self._targettype


    @property
    def tforces(self):
        '''Defines property, providing hint whether atomic forces are present.

        Returns:

            tforces (bool): true, if atomic forces are supplied

        '''

        return self._tforces


    @property
    def predictions(self):
        '''Defines property, providing the predictions of Fortnet.

        Returns:

            predictions (list or 2darray): predictions of the network

        '''

        with h5py.File(self._fname, 'r') as fnetoutfile:
            output = fnetoutfile['fnetout']['output']
            if self._targettype == 'atomic':
                predictions = []
                for idata in range(self._ndatapoints):
                    dataname = 'datapoint' + str(idata + 1)
                    if self._mode == 'validate':
                        predictions.append(
                            np.array(output[dataname]['output'],
                                     dtype=float)[:, :self._npredictions])
                    else:
                        predictions.append(
                            np.array(output[dataname]['output'], dtype=float))
            else:
                predictions = np.empty(
                    (self._ndatapoints, self._npredictions), dtype=float)
                for idata in range(self._ndatapoints):
                    dataname = 'datapoint' + str(idata + 1)
                    if self._mode == 'validate':
                        predictions[idata, :] = \
                            np.array(output[dataname]['output'],
                                     dtype=float)[0, :self._npredictions]
                    else:
                        predictions[idata, :] = \
                            np.array(output[dataname]['output'],
                                     dtype=float)[0, :]

        return predictions


    @property
    def targets(self):
        '''Defines property, providing the targets during training.

        Returns:

            targets (list or 2darray): targets during training

        '''

        if self._mode == 'predict':
            return None

        with h5py.File(self._fname, 'r') as fnetoutfile:
            output = fnetoutfile['fnetout']['output']
            if self._targettype == 'atomic':
                targets = []
                for idata in range(self._ndatapoints):
                    dataname = 'datapoint' + str(idata + 1)
                    targets.append(
                        np.array(output[dataname]['output'],
                                 dtype=float)[:, self._npredictions:])
            else:
                targets = np.empty(
                    (self._ndatapoints, self._npredictions), dtype=float)
                for idata in range(self._ndatapoints):
                    dataname = 'datapoint' + str(idata + 1)
                    targets[idata, :] = \
                        np.array(output[dataname]['output'],
                                 dtype=float)[0, self._npredictions:]

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

        if self._targettype == 'atomic':
            msg = "Error while extracting forces from fnetout file '" \
                + self._fname + \
                "'. Forces only supplied for global property targets."
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
