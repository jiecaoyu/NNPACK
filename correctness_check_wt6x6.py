from __future__ import absolute_import, division, generators, unicode_literals, print_function, nested_scopes, with_statement

########################################################
# script for checking the correctness of conv_wt6x6
########################################################

import subprocess

# Run ninja
subprocess.call('ninja', shell=True)


# Run the bin/correctness-check-wt6x6
## example : ./bin/correctness-check-wt6x6  -ic 9 -oc 16 -is 14 14 -ks 2 2  -m inference -i 40
test_pool = [
        [ 2, 3, 4, 4, 3, 3, 1, 'wt8x8', 'compute'],
        [ 2, 3, 4, 4, 3, 3, 1, 'wt8x8', 'precompute'],
        [ 131, 41, 31, 32, 3, 3, 1, 'wt8x8', 'precompute'],
        ]
for setting in test_pool:
    inst = './bin/correctness-check-wt6x6'
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
