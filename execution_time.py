#Execution time counter

s = time.time()
print('Execution Start')

#insert code here

print ('Execution time: ' + str(round(time.time() - s,3)) + ' seconds (' + str(round((time.time() - s)/60,3)) + ' minutes)' )
