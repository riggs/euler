def nth_prime(nth=10001):
    if nth == 1:
        return 2
    primes = [3]
    n = 5
    while len(primes) < nth-1:
        p = 0
        while primes[p]**2 <= n:
            if n % primes[p] == 0:
                break
            p += 1
        else:
            primes.append(n)
        n += 2
    return primes[-1]
