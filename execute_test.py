from __future__ import absolute_import, division, generators, unicode_literals, print_function, nested_scopes, with_statement

############################
# script for running test
############################

import subprocess

# Run ninja
subprocess.call('ninja', shell=True)


# Run the convolution-benchmark
## example : ./bin/convolution-benchmark  -ic 9 -oc 16 -is 14 14 -ks 2 2  -m inference -i 40
test_pool = [
        [ 256, 256, 32, 32, 3, 3, 50, 'implicit-gemm', 'compute'],
        [ 256, 256, 32, 32, 3, 3, 50, 'wt8x8', 'compute'],
        [ 256, 256, 32, 32, 3, 3, 50, 'wt8x8', 'precompute'],
        ]
for setting in test_pool:
    inst = './bin/convolution-benchmark'
    inst += ' -ic ' + str(setting[0])
    inst += ' -oc ' + str(setting[1])
    inst += ' -is ' + str(setting[2]) + ' ' + str(setting[3])
    inst += ' -ks ' + str(setting[4]) + ' ' + str(setting[5])
    inst += ' -m inference'
    inst += ' -i ' + str(setting[6])
    inst += ' -a ' + setting[7]
    inst += ' -ts ' + setting[8]
    print('=' * 50)
    print(inst)
    subprocess.call(inst, shell=True)
