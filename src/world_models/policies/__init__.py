def proportional_derivative(state, goal):
    return (3 * (goal - state[:2]) - 2 * state[2:]).clip(-1, 1)
