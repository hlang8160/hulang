#!/usr/bin/env python
# -*- coding: UTF-8 -*- 

import argparse
 
def parse_args():
#     description = usage: %prog [options] poetry-file
 
# This is the Slow Poetry Server, blocking edition.
# Run it like this:
 
#   python slowpoetry.py <path-to-poetry-file>
 
# If you are in the base directory of the twisted-intro package,
# you could run it like this:
 
#   python blocking-server/slowpoetry.py poetry/ecstasy.txt
 
# to serve up John Donne's Ecstasy, which I know you want to do.
 
 
    parser = argparse.ArgumentParser()
     
    help = "The addresses to connect."
    parser.add_argument('addresses',nargs = '*',help = help)
 
    help = "The filename to operate on.Default is poetry/ecstasy.txt"
    parser.add_argument('filename',help=help)
 
    help = "The port to listen on. Default to a random available port."
    parser.add_argument('-p','--port', type=int, help=help)
 
    help = "The interface to listen on. Default is localhost."
    parser.add_argument('--iface', help=help, default='localhost')
 
    help = "The number of seconds between sending bytes."
    parser.add_argument('--delay', type=float, help=help, default=.7)
 
    help = "The number of bytes to send at a time."
    parser.add_argument('--bytes', type=int, help=help, default=10)
 
    args = parser.parse_args();
    return args
 
if __name__ == '__main__':
    args = parse_args()
     
    for address in args.addresses:
        print 'The address is : %s .' % address
     
    print 'The filename is : %s .' % args.filename
    print 'The port is : %d.' % args.port
    print 'The interface is : %s.' % args.iface
    print 'The number of seconds between sending bytes : %f'% args.delay
    print 'The number of bytes to send at a time : %d.' % args.bytes#</path-to-poetry-file>