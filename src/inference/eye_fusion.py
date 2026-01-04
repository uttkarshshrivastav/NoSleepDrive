

# inference/eye_fusion.py

def fuse_eye_signals(left, right):
    #combining the signals from both models 
    
  
    # if both are missing
    if left is None and right is None:
        return None
    
    # if only one eye is detected
    if left is None:
        return right
    if right is None:
        return left

    # Both eyes detected
    prob_left = left["prob"]
    prob_right = right["prob"]

    # if both eyes show high closure probability
    if prob_left > 0.3 and prob_right > 0.3:
        #take the higher probality of closure for more safety
        prob = max(prob_left, prob_right)
    elif prob_left > 0.3 or prob_right > 0.3:
        # Only one eye high(doubt) reduce the signal
        avg = (prob_left + prob_right) / 2.0
        prob = avg * 0.7
    else:
        #both inn doubt take average of the signal 
        prob = (prob_left + prob_right) / 2.0

    conf = max(left["conf"], right["conf"])

    return {
        "prob": prob,
        "conf": conf
    }