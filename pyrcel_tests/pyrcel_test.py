import pyrcel as pm
import matplotlib.pyplot as plt
import numpy as np
# Initial conditions

P0 = 92000
T0 = 27
S0 = 0

sea_salt = pm.AerosolSpecies("sea_salt", pm.Lognorm(mu=0.85, sigma=1.2, N=10), kappa=1.2, bins=200)

aerosol = [sea_salt, ]
V = np.logspace(1,3,20)  # Updraft speed (m/s)
fig_T_re,ax_T_re = plt.subplots()
fig_T_z,ax_T_z = plt.subplots()

for v0 in V:
    dt = 1
    Z = 5e3
    t = Z/v0
    model = pm.ParcelModel(aerosol, v0, T0, S0, P0, console=False, accom=0.3)
    parcel_trace, aerosol_trac = model.run(t, dt, solver="cvode")

    def calc_effective_radius(df):
        eff = []
        for row in df.iterrows():
            r3 = sum((r[1]**3 for r in row[1].iteritems()))
            r2 = sum((r[1]**2 for r in row[1].iteritems()))
            eff.append(r3/r2)
        return np.array(eff)


    T = parcel_trace["T"].values
    effective_radius = calc_effective_radius(aerosol_trac["sea_salt"])
    ax_T_re.plot(effective_radius*1e6,T)
    ax_T_re.invert_yaxis()
    ax_T_re.set_xlabel("Effective radius (10^-6 m)")
    ax_T_re.set_ylabel("Temperature (K)")
    ax_T_z.plot(T,parcel_trace["z"].values)
    ax_T_z.set_xlabel("Temperature (K)")
    ax_T_z.set_ylabel("Height (m)")


plt.show()
