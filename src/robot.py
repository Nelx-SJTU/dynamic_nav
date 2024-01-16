import numpy as np

def select_action(env, q, type="random"):
    if type == "random":
        return env.action_space.sample(), " "
    elif type == "keyboard":
        if q == ord("w"):
            return np.array([0.7, 0.0]), " "
        elif q == ord("a"):
            return np.array([0.0, -0.2]), " "
        elif q == ord("d"):
            return np.array([0.0, 0.2]), " "
        elif q == ord("s"):
            return np.array([-0.7, 0.0]), " "
        elif q == ord(" "):
            return np.array([0.0, 0.0]), " "

        # open and close the door
        elif q == ord("o"):
            return np.array([0.0, 0.0]), "open"
        elif q == ord("p"):
            return np.array([0.0, 0.0]), "close"

        # (After topological structure changes)
        # save the partial map to memory bank compulsively
        elif q == ord("k"):
            return np.array([0.0, 0.0]), "save_compulsively"
        elif q == ord("l"):
            return np.array([0.0, 0.0]), "extract_compulsively"

        # start auto save or auto extract
        elif q == ord("n"):
            return np.array([0.0, 0.0]), "auto_save"
        elif q == ord("m"):
            return np.array([0.0, 0.0]), "auto_extract"

    elif type == "auto":
        return np.array([-0.7, 0.0]), " "
