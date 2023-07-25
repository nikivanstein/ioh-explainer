"""
Utility functions
"""

def runParallelFunction(runFunction, arguments, maxJobs = 50):
    """
        Return the output of runFunction for each set of arguments,
        making use of as much parallelization as possible on this system

        :param runFunction: The function that can be executed in parallel
        :param arguments:   List of tuples, where each tuple are the arguments
                            to pass to the function
        :param maxJobs:     Number of threads to start maximum at once.
        :return:
    """
    

    arguments = list(arguments)
    p = Pool(min(maxJobs, len(arguments)))
    results = p.map(runFunction, arguments)
    p.close()
    return results