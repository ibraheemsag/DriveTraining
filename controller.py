import pystk
import numpy as np


def control(aim_point, current_vel, steer_gain=6, skid_thresh=0.2, target_vel=25):
    #this seems to initialize an object
    action = pystk.Action()
    steer_const = 1
    accel_const = 3
    drift_const = 0.25
    if aim_point[0] < 0:
        action.steer = -1 * steer_const
    else:
        action.steer = steer_const

    if current_vel < target_vel:
        action.acceleration = min(1, accel_const * (1 - (current_vel/target_vel)))
        if current_vel - target_vel < -5.5:
            action.nitro = True
    
    else:
        action.nitro = False
        action.acceleration = 0
        action.brake = True
    if aim_point[0] < -1 * drift_const:
        action.drift = True
    if aim_point[0] > drift_const:
        action.drift = True


        

    

    return action

    




if __name__ == '__main__':
    from utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_controller(args)
