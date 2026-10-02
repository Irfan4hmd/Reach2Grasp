from numpy import floor


def set_phase_budgets(success_reports):
    """
    Set phase budgets based on the maximum number of steps taken in each phase across all successful trials.
    """
    phase_steps = {}
    successful_trials = [report for report in success_reports if report.successful]

    if not successful_trials:
        print("No successful trials to calculate phase budgets.")
        return {}

    # Calculate the maximum number of steps for each phase
    for report in successful_trials:
        for phase, steps in report.phase_steps.items():
            if phase not in phase_steps:
                phase_steps[phase] = []
            phase_steps[phase].append(steps)

    # Compute the maximum steps for each phase and set the phase budgets to 2x the average steps and 2 floating point numbers
    phase_budgets = {phase: (max(steps) * 3) for phase, steps in phase_steps.items()}

    return phase_budgets