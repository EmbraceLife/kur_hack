"""
Copyright 2016 Deepgram

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import json
import signal
import atexit
import time
import sys
import argparse
import logging

from . import __version__, __homepage__
from .utils import logcolor
from . import Kurfile
from .engine import JinjaEngine

# get logger a name for display
logger = logging.getLogger(__name__)

###############################################################################
#### create a kurfile object from kurfile.yml and parse it
def parse_kurfile(filename, engine):
	""" Parses a Kurfile.

		# Arguments

		filename: str. The path to the Kurfile to load.

		# Return value

		Kurfile instance
	"""
	## spec is a kurfile object
	spec = Kurfile(filename, engine)
	# logger.warning("spec before parse(): %s", spec.data)

	#### Add stack and structure to spec.data
	## without it error msg: find no model
	spec.parse()
	# logger.warning('spec after parse(): %s', spec.data)
	return spec

###############################################################################
def dump(args):
	""" Dumps the Kurfile to stdout as a JSON blob.
	"""
	### parse kurfile.yml into parts to be used in python code
	spec = parse_kurfile(args.kurfile, args.engine)


	### spec.data store all info from kurfile.yml
	### print out spec.data as json dictionary
	# import json; json.dumps?
    # sort_keys=True, for better organisation
	print(json.dumps(spec.data, sort_keys=True, indent=4))

###############################################################################
def train(args):
	""" Trains a model.
	"""
	### parse kurfile.yml into parts to be used in python code
	spec = parse_kurfile(args.kurfile, args.engine)

	### spec as Kurfile object has a func: get_training_function()
	func = spec.get_training_function()
	func(step=args.step)

###############################################################################
def test(args):
	""" Tests a model.
	"""

	spec = parse_kurfile(args.kurfile, args.engine)
	func = spec.get_testing_function()
	func(step=args.step)

###############################################################################
def evaluate(args):
	""" Evaluates a model.
	"""
	spec = parse_kurfile(args.kurfile, args.engine)
	func = spec.get_evaluation_function()
	func(step=args.step)

###############################################################################
def build(args):
	""" Builds a model.
	"""

	logger.info("Executing build(args) ...  ")

	logger.info("parse kurfile to spec ...  ")
	spec = parse_kurfile(args.kurfile, args.engine)



	### if build's args.compile == auto,
	## but if no train, test, evaluate sections available, build bare model
	## or if train, test, or evaluate sections are available, but data attr is not available, then build a model without data
	if args.compile == 'auto':

		# create an empty list
		result = []
		# if train, test, evalute sections are available in kurfile,
		for section in ('train', 'test', 'evaluate'):
			if section in spec.data:
				# then store the sections inside result list
				result.append((section, 'data' in spec.data[section]))

		logger.info("Since compile == auto, Let result stores: %s", result)

		## but if no train, test, evaluate sections available, build bare model
		if not result:
			# display message
			logger.info('Since result is %s, Set compile to `none`, and Trying to build a bare model.', result)
			# set build complile option to be none
			args.compile = 'none'

		## or if train, test, or evaluate sections are available, but data attr is not available, then build a model without data
		else:
			logger.info("Since result is Not None, but : %s, So ...", result)
			args.compile, has_data = sorted(result, key=lambda x: not x[1])[0]
			logger.info("Set args.compile and has_data to be: %s", sorted(result, key=lambda x: not x[1])[0])

			logger.info('So, trying to build a "%s" model.', args.compile)
			if not has_data:
				logger.info('Since has_data == False, There is not data defined for this model, '
					'though, so we will be running as if --bare was '
					'specified.')

	### if args.compile == none from console, then build a bare model
	elif args.compile == 'none':
		logger.info('Since args.compile == none, Trying to build a bare model.')

	### If args.compile != auto or none, then build a proper model specifically for args.compile == train, or evaluate, or test ...
	else:
		logger.info('Since args.compile == %s, Trying to build a "%s" model.', args.compile, args.compile)

	### If args.bare == True or args.compile == none, build a bare model by setting provider = None
	if args.bare or args.compile == 'none':
		logger.info("Since args.bare or args.compile == none, Set provider None")
		provider = None

	### If arges.bare ==False and args.compile != none, create provider with spec.get_provider(args.compile == train or test, or evaluate)
	else:
		logger.warning("Creates the provider corresponding to a part of the Kurfile == provider")
		provider = spec.get_provider(args.compile)


	logger.warning("Now, spec.get_model(provider) create a spec.model for spec ")

	### with provider ready, either provider == None, or something data, then create spec.model using spec.get_model()
	### spec.model = spec.get_model(provider)
	### Returns the parsed Model instance with data if available
	spec.get_model(provider)

	### with a bare model without data, Build an empty model and exit
	if args.compile == 'none':
		logger.info("Since args.compile == none, Let return nothing")
		return

	### If args.compile == train,
	elif args.compile == 'train':

		### Creates a new Trainer from the Kurfile.
		## create a new loss from Kurfile
		## create a new optimizer from kurfile
		### Then kur.model.Executor() execute the trainer using the 3 components above
		### but without using data
		target = spec.get_trainer(with_optimizer=True)


	### If specify arg.compile == test, build a trainer without optimizer for testing
	elif args.compile == 'test':
		target = spec.get_trainer(with_optimizer=False)
		logger.info("Now target == %s", target)

	### If specify args.compile == evaluate, build a evaluator for evaluation
	elif args.compile == 'evaluate':
		target = spec.get_evaluator()

	### Since args.compile != none, train, evaluate, test, Let return an error message, and exit as failure
	else:
		logger.error('Unhandled compilation target: %s. This is a bug.',
			args.compile)
		return 1

	logger.info("Let's compile target")

	#### target is Executor(trainer or evaluator), target.compile() is to compile the model
	### This generates a backend-specific representation of the model, suitable for training with optimizer or for testing without optimizer
	# without merge data into the model
	target.compile()

###############################################################################
####
def prepare_data(args):
	""" Prepares a model's data provider.
	"""

	### parse kurfile.yml to spec
	spec = parse_kurfile(args.kurfile, args.engine)

	### If args.target == auto,
	### and one of train, validate, test, evaluate is found in spec.data
	### and data is found inside spec.data[section]
	### then set result == the first available of the four above
	if args.target == 'auto':
		result = None
		for section in ('train', 'validate', 'test', 'evaluate'):
			if section in spec.data and 'data' in spec.data[section]:
				result = section
				break

		### If none of train, validate, test, evaluate are available
		### return a value error
		if result is None:
			raise ValueError('No data sections were found in the Kurfile.')
		### set args.target == result, which is None
		args.target = result

	logger.info('Preparing data sources for: %s', args.target)

	### This is to prepare data source for a section available first above
	provider = spec.get_provider(args.target)

	### if args.assemble == True, parse the model with data
	if args.assemble:

		spec.get_model(provider)

		### If args.target == train, create a trainer for it
		if args.target == 'train':
			target = spec.get_trainer(with_optimizer=True)

		### If args.target == test, create a trainer without optimizer for it
		elif args.target == 'test':
			target = spec.get_trainer(with_optimizer=False)

		### If args.target == evaluate, create a evaluator for it
		elif args.target == 'evaluate':
			target = spec.get_evaluator()

		### If none available, error msg: unknow target for assembly? exit as failure
		else:
			logger.error('Unhandled assembly target: %s. This is a bug.',
				args.target)
			return 1

		### Compile the trainer, or evaluator
		target.compile(assemble_only=True)

	#### set batch == None,
	batch = None

	#### get the first batch of the data provider, assign to batch
	for batch in provider:
		break


	#### If batch == None, log error, and exit failure
	if batch is None:
		logger.error('No batches were produced.')
		return 1

	#### Set num_entries = None
	num_entries = None
	### batch is a dictionary, get all of the keys and sorted into a lit
	keys = sorted(batch.keys())
	logger.warning("all the keys of the first batch(dictionary) of data provider: %s", keys)

	#### What is the shape of each key's values
	batch_shape = {}
	for k in keys:
		batch_shape[k] = batch[k].shape
	logger.info("A single batch consists of: %s", batch_shape)

	### count inside the first element of batch dictionary, assign to num_entries
	num_entries = len(batch[keys[0]])
	logger.info("There are total %s samples in a single batch.", num_entries)
	logger.info("Let's have a look a single sample")

	### print out the whole batch, but one sample pair at a time, each sample pair like (key1:value, key2: value)
	# for entry in range(0, num_entries)
	### Here only print out the first and last sample of a batch
	# for entry in range(0, num_entries, num_entries-1):
	for entry in range(1):
		## print Entry entry_idx/total_entries
		print('Entry {}/{}:'.format(entry+1, num_entries))
		## for every batch_key
		for k in keys:
			# print: batch_key: this batch's first item
			print('  {}: {}'.format(k, batch[k][entry]))


	#### If there is no items in a batch, exit with error message
	if num_entries is None:
		logger.error('No data sources was produced.')
		return 1

###############################################################################
def version(args):							# pylint: disable=unused-argument
	""" Prints the Kur version and exits.
	"""
	logger.info("version(args) is running ...")
	print('Kur, by Deepgram -- deep learning made easy')
	print('Version: {}'.format(__version__))
	print('Homepage: {}'.format(__homepage__))

###############################################################################
def do_monitor(args):
	""" Handle "monitor" mode.
	"""

	# If we aren't running in monitor mode, then we are done.
	if not args.monitor:
		return

	# This is the main retry loop.
	while True:
		# Fork the process.
		logger.info('Forking child process.')
		pid = os.fork()

		# If we are the child, leave this function and work.
		if pid == 0:
			logger.info('We are a newly spawned child process.')
			return

		logger.info('Child process spawned: %d', pid)

		# Wait for the child to die. If we die first, kill the child.
		atexit.register(kill_process, pid)
		try:
			_, exit_status = os.waitpid(pid, 0)
		except KeyboardInterrupt:
			break
		atexit.unregister(kill_process)

		# Process the exit code.
		signal_number = exit_status & 0xFF
		exit_code = (exit_status >> 8) & 0xFF
		core_dump = bool(0x80 & signal_number)

		if signal_number == 0:
			logger.info('Child process exited with exit code: %d.', exit_code)
		else:
			logger.info('Child process exited with signal %d (core dump: %s).',
				signal_number, core_dump)

		retry = False
		if os.WIFSIGNALED(exit_status):
			if os.WTERMSIG(exit_status) == signal.SIGSEGV:
				logger.error('Child process seg faulted.')
				retry = True

		if not retry:
			break

	sys.exit(0)

###############################################################################
def kill_process(pid):
	""" Kills a child process by PID.
	"""

	# Maximum time we wait (in seconds) before we send SIGKILL.
	max_timeout = 60

	# Terminate child process
	logger.info('Sending Ctrl+C to the child process %d', pid)
	os.kill(pid, signal.SIGINT)

	start = time.time()
	while True:
		now = time.time()

		# Check the result.
		result = os.waitpid(pid, os.WNOHANG)
		if result != (0, 0):
			# The child process is dead.
			break

		# Check the timeout.
		if now - start > max_timeout:
			# We've waited too long.
			os.kill(pid, signal.SIGKILL)
			break

		# Keep waiting.
		logger.debug('Waiting patiently...')
		time.sleep(0.5)

###############################################################################
def parse_args():
	""" Constructs an argument parser and returns the parsed arguments.
	"""
	parser = argparse.ArgumentParser(
		description='Descriptive deep learning')
	parser.add_argument('--no-color', action='store_true',
		help='Disable colorful logging.')
	parser.add_argument('-v', '--verbose', default=0, action='count',
		help='Increase verbosity. Can be specified twice for debug-level '
			'output.')
	parser.add_argument('--monitor', action='store_true',
		help='Run Kur in monitor mode, which tries to recover from critical '
			'errors, like segmentation faults.')
	parser.add_argument('--version', action='store_true',
		help='Display version and exit.')

	subparsers = parser.add_subparsers(dest='cmd', help='Sub-command help.')

	subparser = subparsers.add_parser('train', help='Trains a model.')
	subparser.add_argument('--step', action='store_true',
		help='Interactive debug; prompt user before submitting each batch.')
	subparser.add_argument('kurfile', help='The Kurfile to use.')
	### assign args.func = train()
	subparser.set_defaults(func=train)

	subparser = subparsers.add_parser('test', help='Tests a model.')
	subparser.add_argument('--step', action='store_true',
		help='Interactive debug; prompt user before submitting each batch.')
	subparser.add_argument('kurfile', help='The Kurfile to use.')
	### assign args.func = test()
	subparser.set_defaults(func=test)

	subparser = subparsers.add_parser('evaluate', help='Evaluates a model.')
	subparser.add_argument('--step', action='store_true',
		help='Interactive debug; prompt user before submitting each batch.')
	subparser.add_argument('kurfile', help='The Kurfile to use.')
	# assign args.func = evalute()
	subparser.set_defaults(func=evaluate)

	subparser = subparsers.add_parser('build',
		help='Tries to build a model. This is useful for debugging a model.')
	subparser.add_argument('kurfile', help='The Kurfile to use.')
	subparser.add_argument('-c', '--compile',
		choices=['none', 'train', 'test', 'evaluate', 'auto'], default='auto',
		help='Try to compile the specified variation of the model. If '
			'--compile=none, then it only tries to assemble the model, not '
			'compile anything. --compile=none implies --bare')
	subparser.add_argument('-b', '--bare', action='store_true',
		help='Do not attempt to load the data providers. In order for your '
			'model to build correctly with this option, you will need to '
			'specify shapes for all of your inputs.')
	### assign args.func = build()
	subparser.set_defaults(func=build)

	subparser = subparsers.add_parser('dump',
		help='Dumps the Kurfile out as a JSON blob. Useful for debugging.')
	subparser.add_argument('kurfile', help='The Kurfile to use.')
	### assign args.func = dump()
	subparser.set_defaults(func=dump)

	subparser = subparsers.add_parser('data',
		help='Does not actually compile anything, but only prints out a '
			'single batch of data. This is useful for debugging data sources.')
	subparser.add_argument('kurfile', help='The Kurfile to use.')
	subparser.add_argument('-t', '--target',
		choices=['train', 'validate', 'test', 'evaluate', 'auto'],
		default='auto', help='Try to produce data corresponding to a specific '
			'variation of the model.')
	subparser.add_argument('--assemble', action='store_true', help='Also '
		'begin assembling the model to pull in compile-time, auxiliary data '
		'sources.')
	### assign args.func = prepare_data()
	subparser.set_defaults(func=prepare_data)

	return parser.parse_args()

###############################################################################
#### This is where everything start to run
def main():
	""" Entry point for the Kur command-line script.
	"""


	### Take args from console to this main()
	args = parse_args()



	### set 3 levels for logging display:
	loglevel = {
		0 : logging.WARNING,  # 0: default, warning
		1 : logging.INFO,     # 1: info
		2 : logging.DEBUG     # 2: debug
	}


	### Use basic logging configure or custom color logging configure
	config = logging.basicConfig if args.no_color else logcolor.basicConfig

	### Let configure logging display
	config(

		## use args.verbose to select loglevel
		# check: d = {}; d.get?
		level=loglevel.get(args.verbose, logging.DEBUG),

		## format logging display:
		# distance: \n \n
		# logging color: {color} and {reset}
		# logging address: [%(levelname)s %(asctime)s %(name)s:%(lineno)s]
		format='{color}[%(levelname)s %(asctime)s %(name)s:%(lineno)s]{reset} '
			'%(message)s'.format(
				color='' if args.no_color else '$COLOR',
				reset='' if args.no_color else '$RESET'
			)
	)


	logger.warning("console args: %s", args)


	### pass warning message to logging package
	# import logging; logging.captureWarnings?
	logging.captureWarnings(True)



	### args control when to run do_monitor()
	do_monitor(args)


	### args.version controls: args.func = version
	if args.version:
		# set args.func = version, version is a function defined above
		args.func = version

	### `kur -v`: has no func attribute
	elif not hasattr(args, 'func'):
		logger.info("You ask for nothing, so ... ... ")
		## import sys; sys.stderr?
		print('Nothing to do!', file=sys.stderr)
		print('For usage information, try: kur --help', file=sys.stderr)
		print('Or visit our homepage: {}'.format(__homepage__))
		## import sys; sys.exit?
		# exit as failure?
		sys.exit(1)

	# import kur; from kur.engine import JinjaEngine; JinjaEngine?
	# Creates a new Jinja2 templating engine.
	logger.info("Create a JinjiaEngine object ... ... ")
	engine = JinjaEngine()
	# engine is needed for making Kurfile() object


	### assing a new Jinja2 templating engine to args.engine
	setattr(args, 'engine', engine)


	### Let's run the core function to perform and exit system
	logger.warning("Let's executing %s(args) ... ... ", args.func)
	sys.exit(args.func(args) or 0)

	### args.func are defined inside parse_args(), go up for details
###############################################################################

#### We are able to run this __main__.py from console
if __name__ == '__main__':
	main()

### EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF.EOF
