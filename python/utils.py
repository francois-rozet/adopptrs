"""
"""

###########
# Imports #
###########

import os
import os.path


#############
# Functions #
#############

def list_files(dirname, ext=None):
	'''Lists recursively all files in @dirname.'''
	l = list()

	for root, _, files in os.walk(dirname):
		for f in files:
			if ext is None or f.endswith(ext):
				l.append(os.path.join(root, f))

	return l
