import numpy as np
import matplotlib.pylab as plt

c = 299792458 # speed of light [m/s]
sampling_frequency, fc = 1e10, 3e8
# sampling frequency, chip frequency
n_ref, m = 2, 1 # reflection order, sequence repetition
lines_len = np.array([2, 1, 2.3]) # lines lengths [m]
attenuation = 0 # lines attenuation [dB/m]
Zc, Zl1, Zl2 = 75, 0, 0
velocity_factor = 0.71

sn = np.array([ 1,  1,  1,  1,  1,  1, -1,  1, -1,  1,
    -1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1, -1,  1,
    1, -1,  1, -1, -1,  1, -1, -1,  1,  1,  1, -1, -1, -1,
    1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1, -1, -1, -1,
    1,  1, -1, -1, -1, -1,  1, -1, -1, -1, -1, -1,])
# m-sequence of length 63

def loads_impedances(Zl1, Zl2) :
    Zl = np.array([Zl1, Zl2], dtype = 'complex')
    if sum(Zl) == 0 :
        Zl = Zl+1e-6
    return Zl
Zl = loads_impedances(Zl1, Zl2)

propagation_velocity = velocity_factor*c 
# propagation velocity [m/s]
Tn, Tc = 1/sampling_frequency, 1/fc
# sampling period, chip duration

def domains (n_ref, m) :
    Ts = len(sn)*Tc # duration of sequance 
    Tsm = m*Ts # duration of m times the sequence
    tmax = Tsm+(
        (n_ref*sum(lines_len))/
        propagation_velocity)
    time = np.arange(0, tmax-Tn, Tn) # time base [s]
    nSamples = len(time)
    frequency = np.fft.fftfreq(nSamples, Tn) 
    # frequency vector [Hz]
    return time, frequency
t, f = domains (n_ref, m) # time and frequency vectors
distance = (t*propagation_velocity)/2 # distance vector

omega = 2*np.pi*f # angular frequency [rad/sec]
alpha = attenuation/8.68 # lines attenuation [Np/m]
beta = omega/propagation_velocity
# phase constant of the lines [rad/m]
gamma = alpha+1j*beta # complex propagation constant

def Rin(Zc,Zl) :
    Rl = np.zeros(len(Zl), dtype = 'complex')
    for i in range(len(Zl)) :
        Rl[i] = (Zl[i]-Zc)/(Zl[i]+Zc)
        # reflection coeff at the loads
    R = np.zeros((len(Rl), len(gamma)), dtype = 'complex')
    for i in range(len(lines_len)-1) :
        R[i,:] = Rl[i]*np.exp(-2*gamma*lines_len[i+1])
        # reflection coeff at origin of each branch
        R0 = ((-1+R[0,:]+R[1,:]+3*R[0,:]*R[1,:])/
            (3+R[0,:]+R[1,:]-R[0,:]*R[1,:]))
        # reflection coeff at end of injection line
        Rin = R0*np.exp(-2*gamma*lines_len[0])
        # reflection coeff at injection port
    return Rin # reflection coeff at injection port

def fourier_transform(signal):
    # fast fourier transform
    return np.fft.fft(signal)

def inverse_fourier_transform(signal):
    # inverse fast fourier transform
    return np.fft.ifft(signal)

def correlation(reference, signal) :
    return np.correlate(reference, signal, mode="same")

def sequence(sn) :
    p = np.ones(len(np.arange(0, Tc-Tn, Tn)))
    # pulse of duration = to Tc
    sequence = sn[1]*p
    i = np.arange(1, m*len(sn), 1)
    index = np.mod(i, len(sn))
    # index used to generate the sequence
    for i in np.arange(0, m*len(sn)-1, 1) :
        si = np.concatenate((sequence, sn[index[i]]*p))
        sequence = si
    return sequence
stdr = sequence(sn)
stdr.resize(len(t), refcheck = False)
sstdr = stdr*np.cos(2*np.pi*fc*t)
STDR = fourier_transform(stdr)
SSTDR = fourier_transform(sstdr)

def gaussian_pulse(sigma) :
    gpulse = np.exp(-(t-4e-9)**2/(2*sigma**2))
    Gpulse = fourier_transform(gpulse)
    return gpulse, Gpulse
sigma = 1e-9
gpulse, Gpulse = gaussian_pulse(sigma)

#### network response to injected signals ####
Uk = STDR*Rin(Zc,Zl)
Vk = SSTDR*Rin(Zc,Zl)
Gk = Gpulse*Rin(Zc,Zl)
uk = inverse_fourier_transform(Uk)
vk = inverse_fourier_transform(Vk)
gk = inverse_fourier_transform(Gk)
###############################################

def noise(p, n_Samples) :
    return np.sqrt(p)*np.random.normal(0, 1, 
        len(n_Samples))
noise = noise(0.5, t)

#### received signals ####
x = uk+noise # received signal x(t)
y = vk+noise # received signal y(t)
##########################

#### signals correlation ####
rux = correlation(stdr, x)
rvy = correlation(sstdr, y)
##############################

#### result ploting ####
fig, ax = plt.subplots()
ax.plot(distance, gk.real)
ax.grid(True)
ax.set_xlabel('distance [m]')
ax.set_ylabel('signal amplitude [V]')
ax.set_xlim(0, 10)
plt.show()
########################
