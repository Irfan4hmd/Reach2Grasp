class PhaseTimeout(Exception):
    """Custom exception to indicate that a phase has exceeded its time budget."""
    def __init__(self, phase_name: str):
        super().__init__(f"timeout:{phase_name}")
        self.phase_name = phase_name