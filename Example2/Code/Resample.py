import numpy as np
import matplotlib.pyplot as plt


base = '../'
stress = np.load(base+'us_valid.npz')
xyzd = np.load(base+'xyzd_valid.npz')

N_case = 2500

for rpt in range(3):
	for N_resample in [ 5000 ]:
		output_pos = np.zeros([N_case,N_resample,4])
		output_vals = np.zeros([N_case,N_resample,4])

		for k in range(N_case):
			name = 'Valid' + str(k)

			my_xyzd = xyzd[name]
			my_stress = stress[name]

			npt = len(my_xyzd)

			# Sample points
			if npt <= N_resample:
				index = np.random.choice( np.arange(npt) , N_resample , replace=True )
			else:
				index = np.random.choice( np.arange(npt) , N_resample , replace=False )

			output_pos[ k , : , : ] = my_xyzd[ index , : ].copy()
			output_vals[ k , : , : ] = my_stress[ index , : ].copy()

		np.savez_compressed('Outputs_rpt'+str(rpt)+'_N'+str(N_resample)+'.npz',a=output_pos,b=output_vals)

		plt.hist( output_vals[:,:,0].flatten() , bins=50 )
		plt.xlabel('vM stress [MPa]')
		plt.savefig('StressHist_rpt'+str(rpt)+'_N'+str(N_resample)+'.pdf')
		plt.close()

		cl = ['','X','Y','Z']
		for cc in [1,2,3]:
			plt.hist( output_vals[:,:,cc].flatten() , bins=50 )
			plt.xlabel('U'+ cl[cc] +' [mm]')
			plt.savefig('U'+ cl[cc] +'_rpt'+str(rpt)+'_N'+str(N_resample)+'.pdf')
			plt.close()