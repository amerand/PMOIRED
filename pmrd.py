#!/usr/bin/env python3

# https://github.com/amerand/PMOIRED

import os, sys, shutil, pathlib

home = os.path.expanduser('~')
pmrd = os.path.join(home, '.pmrd')
github = 'https://github.com/amerand/PMOIRED'

add_packages = ['jupyterlab','ipympl', 'notebook',
				'catppuccin-jupyterlab']

help_text = f"""manage your PMOIRED python environment with {sys.argv[0]}

options:
  --help or -h: this help
  --install: install the environment in {pmrd}, including PMOIRED from github
	{github}
  --update or -u: update to the latest gihub version of PMOIRED
  --update-all: update the major packages:
                numpy scipy matplotlib astropy astroquery jupyterlab
  --version or -v: show the versions of main libraries
  --remove: remove the environment
  --python or -p : start python console or run script
  --ipython or -i : start ipython console
  --notebook or -n : start notebook (can pass a directory of file)
  --jupyter-lab or -j : start jupyter-lab (can pass a directory of file)
  --examples or -e: start jupyterlab with examples from
	{github}_examples

to activate the envirnoment manually, run:

source {pmrd}/bin/activate"""

do_help =  len(sys.argv)==1 or ('--help' in sys.argv or '-h' in sys.argv)

do_install = '--install' in sys.argv
do_update = '--update' in sys.argv or '-u' in sys.argv
do_update_all = '--update-all' in sys.argv
do_remove = '--remove' in sys.argv
do_jlab = '--jupyter-lab' in sys.argv or '-j' in sys.argv
do_note = '--notebook' in sys.argv or '-n' in sys.argv
do_ipy = '--ipython' in sys.argv or '-i' in sys.argv
do_py = '--python' in sys.argv or '-p' in sys.argv
do_ver = '--version' in sys.argv or '-v' in sys.argv
do_expl = '--examples' in sys.argv or '-e' in sys.argv

if len(sys.argv)>1:
	directory = list(filter(lambda x: os.path.exists(os.path.expanduser(x)),
							sys.argv[1:]))
else:
	directory = ''
	
install = f"""python3 -m venv {pmrd}
. {pmrd}/bin/activate
pip install -U pip
pip install git+{github}
pip install {' '.join(add_packages)}"""

prgm = ''
if do_jlab:
	prgm = 'jupyter-lab'
if do_note:
	prgm = 'jupyter-notebook'
if do_ipy:
	prgm = 'ipython'
if do_py:
	prgm = 'python'
if do_ver:
	prgm = 'python'
	
run = f""". {pmrd}/bin/activate
{pmrd}/bin/{prgm} """

if not do_ver: 
	if len(directory)==1:
		run = f""". {pmrd}/bin/activate
cd {os.path.abspath(os.path.dirname(directory[0]))}
{pmrd}/bin/{prgm} """

	if not os.path.isdir(directory[0]):
		run += os.path.basename(directory[0]).replace(' ', '\\ ')
	else:
		run += ' '.join(directory)
else:
	run += ' -c "import pmoired, pprint; pprint.pprint(pmoired.__versions__)"'
	
update_pmoired = f""". {pmrd}/bin/activate
pip install -U git+{github}
"""

update_all = f""". {pmrd}/bin/activate
pip install -U pip numpy scipy matplotlib astropy astroquery
pip install -U pip {' '.join(add_packages)}
pip install -U git+{github}
"""

get_examples = f"""cd {pmrd}
git clone https://github.com/amerand/PMOIRED_examples
"""

update_examples = f"""cd {pmrd}/PMOIRED_examples
git pull
"""

run_examples = f""". {pmrd}/bin/activate
cd {pmrd}/PMOIRED_examples/notebooks
{pmrd}/bin/jupyter-lab 
"""

def main():
	if do_help:
		print(help_text)
		return
		
	if not os.path.exists(pmrd) or do_install :
		print('#'*20+"\n INSTALLING\n"+'#'*20)
		os.system(install)
		return
	
	if not os.path.exists(pmrd):
		print(f'environment {pmrd} does not exist, nothing to do')
		print(f'run "{sys.argv[0]} --install" first ')
		return
	
	if do_expl:
		if not os.path.exists(os.path.join(pmrd, 'PMOIRED_examples')):
			os.system(get_examples)
		os.system(run_examples)

	if do_remove:
		r = input(f'are you sure you want to remove {pmrd}? (y/[n]]: ')
		if r.lower().strip()=='y':
			shutil.rmtree(pmrd)
		return
		
	if do_update:
		print('#'*30+"\n UPDATING PMOIRED from github\n"+'#'*30)
		os.system(update_pmoired)
		if os.path.exists(os.path.join(pmrd, 'PMOIRED_examples')):
			print('#'*30+"\n UPDATING PMOIRED examples\n"+'#'*30)
			os.system(update_examples)
		return

	if do_update_all:
		print('#'*20+"\n UPDATING ALL\n"+'#'*20)
		os.system(update_all)
		return
		
	if prgm!='':
		print(run)
		os.system(run)

if __name__=='__main__':
	main()
