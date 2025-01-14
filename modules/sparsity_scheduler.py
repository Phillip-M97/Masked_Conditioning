# this is the increasing version of the sparsity schedulers
# means the sparsity always stay constant or increases during training

import math as m

from modules.mcvae import mcVAE

import matplotlib.pyplot as plt

class SparsityScheduler():
    ''' Sparsity Scheduler for manipulating the sparsity of the Mask() layer during training.
    '''

    def __init__(self, model: mcVAE, arch: dict, params: dict, spars_params: dict, use_epochs: bool=True) -> None:
        
        self.mask = model.msk

        # StepSpars, ExponentialSpars can't handle target sparsity 0
        if spars_params['target_sparsity'] == 0 and spars_params['sparsity_scheduler'] in [StepSpars, ExponentialSpars]:
            self.target_sparsity = 1e-10 if spars_params['target_sparsity'] == 0 else spars_params['target_sparsity']
        else:
            self.target_sparsity = spars_params['target_sparsity']
        
        # StepSpars, ExponentialSpars, CosineAnnealingSpars can't handle init sparsity 0
        if spars_params['sparsity'] == 0 and spars_params['sparsity_scheduler'] in [StepSpars, ExponentialSpars, CosineAnnealingSpars]:
            self.init_sparsity = model.msk.p = 1e-10 if spars_params['sparsity'] == 0 else spars_params['sparsity']
        else:
            self.init_sparsity = model.msk.p = spars_params['sparsity']

        self.history = [self.init_sparsity]
        self.cur_epoch = 1
        self.last_epoch = params['epochs'] if use_epochs else params['steps']
        if use_epochs:
            self.target_epoch = params['epochs'] if spars_params['target_epoch'] == -1 else spars_params['target_epoch']
        else:
            self.target_epoch = params['steps'] if spars_params['target_steps'] == -1 else spars_params['target_steps']

        assert 0 <= self.init_sparsity <= 1, 'initial sparsity must be in range [0,1]'
        assert 0 <= self.target_sparsity <= 1, 'target sparsity must be in range [0,1]'
        assert (1 <= self.target_epoch <= self.last_epoch) or self.target_epoch == -1, 'implausible target epoch'

    def get_last_sparsity(self) -> float:
        ''' Returns current sparsity of the Mask layer.
        '''
        return self.mask.p
    
    def get_sparsity(self) -> list:
        ''' Returns list of all sparsities over the training.
        '''
        return self.history

    def step(self) -> None:
        self.cur_epoch += 1
        self._step()
        self.history.append(self.mask.p)

    def plot(self) -> None:
        ''' Plots the sparsity over epochs.
        '''
        plt.plot(list(range(self.last_epoch)), self.history)
        plt.show()

class ConstantSpars(SparsityScheduler):
    ''' Sparsity stays constant over all epochs.
    '''

    def __init__(self, model:mcVAE, arch: dict, params: dict, spars_params: dict, use_epochs: bool=True) -> None:
        super().__init__(model, arch, params, spars_params, use_epochs)

    def __name__(self):
        return 'ConstantSpars'
        
    def _step(self):
        pass

class StepSpars(SparsityScheduler):
    ''' Increases the sparsity by factor gamma every step_size such that the sparsity achieves the target_sparsity at target_epoch.
    '''
    
    def __init__(self, model: mcVAE, arch: dict, params: dict, spars_params: dict, use_epochs: bool=True) -> None:
        super().__init__(model, arch, params, spars_params, use_epochs)

        self.step_size = spars_params['step_size']

        assert 1 <= self.step_size <= self.last_epoch, 'step size must be smaller than last epoch'
        
        if self.init_sparsity < self.target_sparsity:
            self.gamma = m.e ** (m.log(self.target_sparsity / self.init_sparsity) / (self.target_epoch / self.step_size - 1))
        elif self.init_sparsity > self.target_sparsity:
            self.gamma = (self.target_sparsity / self.init_sparsity) ** (1 / ((self.target_epoch / self.step_size) - 1))
        else:
            self.gamma = 1

    def __name__(self):
        return 'StepSpars'

    def _step(self) -> None:
        if self.cur_epoch % self.step_size == 0:
            new_p = self.mask.p * self.gamma

            if (self.init_sparsity < self.target_sparsity and new_p < self.target_sparsity) or \
                (self.init_sparsity > self.target_sparsity and new_p > self.target_sparsity):
                self.mask.p = new_p
            else:
                self.mask.p = self.target_sparsity

class LinearSpars(SparsityScheduler):
    ''' Increases the sparsity by factor gamma such that the sparsity achieves the target_sparsity at target_epoch in a linear way.
    '''

    def __init__(self, model: mcVAE, arch: dict, params: dict, spars_params: dict, use_epochs: bool=True) -> None:
        super().__init__(model, arch, params, spars_params, use_epochs)

        self.add = abs(self.init_sparsity - self.target_sparsity) / self.target_epoch

        if self.init_sparsity > self.target_sparsity:
            self.add = -self.add

    def __name__(self):
        return 'LinearSpars'

    def _step(self) -> None:
        new_p = self.mask.p + self.add
        
        if (self.init_sparsity < self.target_sparsity and new_p < self.target_sparsity) or \
            (self.init_sparsity > self.target_sparsity and new_p > self.target_sparsity):
            self.mask.p = new_p
        else:
            self.mask.p = self.target_sparsity

class ExponentialSpars(SparsityScheduler):
    ''' Decays the sparsity exponentially until the target_epoch is reached or till the end of training when target_epoch=-1.
    '''

    def __init__(self, model: mcVAE, arch: dict, params: dict, spars_params: dict, use_epochs: bool=True) -> None:
        super().__init__(model, arch, params, spars_params, use_epochs)

        self.gamma = (self.target_sparsity / self.init_sparsity) ** (1 / self.target_epoch)

    def __name__(self):
        return 'ExponentialSpars'

    def _step(self) -> None:
        new_p = self.mask.p * self.gamma
        
        if new_p != self.target_sparsity:
            self.mask.p = new_p
        else:
            self.mask.p = self.target_sparsity

class CosineAnnealingSpars(SparsityScheduler):
    ''' Decays the sparsity with a cosine annealing strategy until the target_epoch is reached or till the end of training when target_epoch=-1.
    '''

    def __init__(self, model: mcVAE, arch: dict, params: dict, spars_params: dict, use_epochs: bool=True) -> None:
        super().__init__(model, arch, params, spars_params, use_epochs)

    def __name__(self):
        return 'CosineAnnealingSpars'

    def _step(self) -> None:
        if self.cur_epoch < self.target_epoch:
            self.mask.p = self.target_sparsity + 0.5 * (self.init_sparsity - self.target_sparsity) * (1 + m.cos(m.pi * self.cur_epoch / self.target_epoch))
        elif self.init_sparsity > self.target_sparsity:
            self.mask.p = self.init_sparsity + 0.5 * (self.target_sparsity - self.init_sparsity) * (1 - m.cos(m.pi * self.cur_epoch / self.target_epoch))
        else:
            self.mask.p = self.target_sparsity

