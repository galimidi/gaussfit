import numpy as np
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize

# function to make the resulting fitted spectrum
def gaussians(x, params):
    y = np.zeros_like(x)  # initialize line
    if len(params) == 0:  # handle empty params (where there are no peaks)
        return y          # if there are no gaussians, just return the continuum
    for i in range(0, len(params), 3):
        amp, mean, std = params[i], params[i+1], params[i+2]  # params variable contains [amplitude, mean, stddev] of the gaussians
        with np.errstate(divide='ignore', invalid='ignore'):  # ignore division by zero errors
            y += amp * np.exp(-0.5 * ((x - mean) / std) ** 2)  # add a gaussian with the above parameters to the line
    return y

# function to be minimized
def negloglikelihood(params, x, y, errors):
    model = gaussians(x, params)  # create the multi-gaussian line
    residuals = y - model
    return 0.5 * np.sum((residuals / errors) ** 2) + np.sum(np.log(np.sqrt(2 * np.pi) * errors))

# calculate AIC and BIC
def info_crit(params, wave, y, errors):
    n, k = len(y), len(params)  # number of data points and parameters
    aic = 2 * k + 2 * negloglikelihood(params, wave, y, errors)  # AIC = 2k - 2*log(L)
    bic = k * np.log(n) + 2 * negloglikelihood(params, wave, y, errors)  # BIC = k*log(n) - 2*log(L)
    return aic, bic

def gaussfit(wave, spectrum, noise, crit='bic', tolerance=1000, maxgauss=15):

    # ========================= peaks stuff =========================
    spectrum -= np.median(spectrum)
    smooth = savgol_filter(spectrum, window_length=11, polyorder=2)
    
    peaks, properties = find_peaks(smooth, height=1.125*noise, prominence=0.9*noise,  # less strict criteria
                                   width=25*np.mean(np.diff(wave)), distance=None)
    
    if len(peaks) == 0:  # for the fits where no gaussians are fitted
        class nothing: x=np.array([]); aic=np.inf; bic=np.inf
        return nothing()
    
    sort_idx = np.argsort(properties['peak_heights'])[::-1]  # sort by descending height
    peaks, widths = peaks[sort_idx], properties['widths'][sort_idx]
    sigmas = np.diff(wave)[0] * widths / (2*(2*np.log(2))**0.5)  # stddev = FWHM / 2*sqrt(2*ln(2))
    
    # ======================== fitting stuff ========================
    models, aics, bics, chisqs = [], [], [], []  # store results for different models
    
    # first, run the null model (no peaks) first to get baseline AIC/BIC
    null_aic, null_bic = info_crit(np.array([]), wave, spectrum, noise)
    models.append((np.array([]), null_aic, null_bic, np.nan))
    aics.append(null_aic); bics.append(null_bic); chisqs.append(0)
    
    # run models with increasing number of peaks
    for n in range(1, min(len(peaks), maxgauss) + 1):  # setting max peaks to 15
        init, bounds = [], []
        for p, sig in zip(peaks[:n], sigmas[:n]):
            amp, mean = spectrum[p], wave[p]
            init.extend([amp, mean, sig])  # initial guesses
            bounds.extend(([amp-50, amp+50], [mean-5, mean+5], [0, np.inf]))
        result = minimize(negloglikelihood, init, args=(wave, spectrum, noise), bounds=bounds, tol=1e-6)

    # ==================== analysis of fit stuff ====================
        aic, bic = info_crit(result.x, wave, spectrum, noise)
        chisq = np.sum(((spectrum-gaussians(wave, result.x))/noise)**2) / (len(spectrum)-len(result.x))
        models.append((result.x, aic, bic, chisq)) 
        aics.append(aic); bics.append(bic); chisqs.append(chisq)

    info = bics if crit == 'bic' else aics  # select criterion
    best_idx = np.where(info <= np.min(info) + tolerance)[0][0]  # best model is first to meet criterion within tolerance

    best = type('Result', (), {
        'x': [models[best_idx][0], models[best_idx][1], models[best_idx][2], models[best_idx][3]],  # parameters, AIC, BIC, chisq of best model
        'all': [models, aics, bics, chisqs]  # parameters, AIC, BIC, chisq of all models                                 
    })()
    
    return best