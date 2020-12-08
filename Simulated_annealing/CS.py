
def cooling_schedule(start_temp, max_iterations, iteration, kind):
    alpha = 1.5
    if kind  == "Linear":
        current_temp = start_temp/(1 + alpha*iteration)
    elif kind == "Linear2":
        current_temp = start_temp/(start_temp/max_iterations) * iteration
    elif kind == "Log":
        current_temp =  start_temp/(1 + alpha * (math.log(start_temp + iteration, 10)))
    elif kind == "Exponential":
        current_temp = start_temp*0.9**iteration
    elif kind == "Quadratic":
        current_temp =  start_temp/(1 + alpha * iteration**2)
    return current_temp