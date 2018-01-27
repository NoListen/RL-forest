# BiCNet

This is a implementation of [BiCNet](https://arxiv.org/pdf/1703.10069.pdf).

`Dynamic` means that applying the dynamic RNN to ignore the dead units.

By setting `au`(alive units) to the numebr of all soldiers in self-camp, dynamci RNN can perform like static RNN.

# Environment

This is compatible with [gym-starcraft](https://github.com/NoListen/gym-starcraft) where I made a lot of changes to the environment.

The observations are represented as dictionary and the keys are the same as the models' parameters of call function.

The actual enviroments need to be adjusted in the SC setting. Gym enviroment only serves as a API and can't determine the environment directly. You can modify the enviroments by creating different soldiers in different camps and positions.

# Notes

Convolution Networks can also be tried which is supported by this framwork and the environment mentioned above.

However, the pixel differences are very small. I didn't get a good result.


