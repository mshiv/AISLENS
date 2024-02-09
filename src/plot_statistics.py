# To plot exploratory statistics of the raw and processed datasets

"""
Functions:
1. scatter: Plot the dependence of melt rate on draft
2. histogram: Plot the histogram of the melt rate
3. time_series: Plot a spatially averaged time series of the melt rate over a defined domain (can be full ice sheet, or specific regions)
4. power_spectrum: Plot the power spectrum of the melt rate time series
5. autocorrelation: Plot the autocorrelation of the melt rate time series

"""

def scatter(melt, draft):
    """
    Plot the dependence of melt rate on draft
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(draft, melt, s=1, c='k')
    plt.xlabel('Draft (m)')
    plt.ylabel('Melt rate (m/yr)')
    plt.title('Melt rate vs. draft')
    plt.show()

def histogram(melt):
    """
    Plot the histogram of the melt rate
    """
    plt.figure(figsize=(8, 6))
    plt.hist(melt, bins=100, color='k')
    plt.xlabel('Melt rate (m/yr)')
    plt.ylabel('Frequency')
    plt.title('Melt rate histogram')
    plt.show()

def time_series(melt, domain):
    """
    Plot a spatially averaged time series of the melt rate over a defined domain
    """
    plt.figure(figsize=(8, 6))
    melt.mean(dim=domain).plot()
    plt.xlabel('Time')
    plt.ylabel('Melt rate (m/yr)')
    plt.title('Melt rate time series')
    plt.show()

def power_spectrum(melt):
    """
    Plot the power spectrum of the melt rate time series
    """
    plt.figure(figsize=(8, 6))
    melt = melt - melt.mean()
    f, Pxx = signal.periodogram(melt, fs=1)
    plt.loglog(f, Pxx, color='k')
    plt.xlabel('Frequency (1/yr)')
    plt.ylabel('Power')
    plt.title('Power spectrum of melt rate')
    plt.show()

def autocorrelation(melt):
    """
    Plot the autocorrelation of the melt rate time series
    """
    plt.figure(figsize=(8, 6))
    ac = signal.correlate(melt, melt, mode='full')
    ac = ac / np.max(ac)
    plt.plot(ac, color='k')
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation of melt rate')
    plt.show()
