import numpy as np
import cv2

def cv2_drawTracks(img, tracks, latestpointcolor, oldpointcolor,
                   plot_last_kp = 40,
                   min_color_gradient = 20
                  ):
    for idx, kp_pts in enumerate(tracks):
        prev_kp = None
        if isinstance(latestpointcolor,np.ndarray):

            color_kp = np.asarray(oldpointcolor)[idx, :]
            colinc = ((np.asarray(latestpointcolor)[idx, :] - color_kp) 
                      / max(min_color_gradient, len(kp_pts)))
        else:
            color_kp = np.asarray(oldpointcolor)
            colinc = ((np.asarray(latestpointcolor) - color_kp) 
                      / max(min_color_gradient, len(kp_pts)))

        # Plot only last 40 key points
        for kp_pt in kp_pts[-plot_last_kp:]:
            (x2, y2) = kp_pt
            if prev_kp is not None:
                # none masked
                (x1, y1) = prev_kp
                cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)),
                     tuple(color_kp), 2)
            else:
                cv2.circle(img, (int(x2), int(y2)), 2, tuple(color_kp), -1)

            prev_kp = kp_pt
            color_kp = color_kp + colinc
        cv2.circle(img, (int(x2), int(y2)), 2, tuple(color_kp), -1)

    return img

