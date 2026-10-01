class SuccessReport:
    def __init__(self):
        self.successful = False
        self.last_phase = None
        self.object_to_target_distance = 0
        self.trial_number = 0
        self.steps = 0
        self.phase_steps = {}
        self.reason = None  # Reason for failure, if any
        
        