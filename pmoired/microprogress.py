import serial, os, time, hashlib, platform

_trusted = ['8eb3b21210d8b26a182b6b3cf2fb817508986826bfa38cb7ea66352640fb91ad']

def _guessPorts():
    if os.uname().sysname == 'Darwin':
        ports = [os.path.join('/dev', x) for x in os.listdir('/dev/')
                if x.startswith('tty') and '.usbmodem' in x]
    return ports

def _getId(port):
    return hashlib.sha256(bytes('|'.join([platform.node(), 
                                   port]), 'utf8')).hexdigest()

def _scan(ports=None, debug=False):
    if ports is None:
        ports = _guessPorts()
        if debug:
            print('dbg>', 'detected ports', ports)
    res = []
    for p in ports:
        print('testing port', p)
        ser = serial.Serial(p, 115200, timeout=1)
        _ = ser.readlines()
        ser.write(b'\x04') # ctrl+d
        time.sleep(1) # give time for board to reset
        r = ser.readlines()
        if not r[1].decode().startswith('MPY'):
            print('not a micropython board!')
            continue
        if debug:
            print('dbg>', 'answer to soft-reboot:', r)
        code = ['import os',
                'os.listdir()']
        ser.write(bytes('\r'.join(code)+'\r', 'utf8'))
        time.sleep(0.1) # give time for board
        r = ser.readlines()
        if debug:
            print('dbg>', 'files on board:', r)
        # -- list of files on board
        l = eval(r[-2][:-2].decode())
        if 'muprgrss.py' in l:
            code = ['import muprgrss', 
                    'muprgrss.test()',
                    ]
            ser.write(bytes('\r'.join(code)+'\r', 'utf8'))
            r = ser.readlines()
            if debug:
                print('dbg>', 'answer to muprgrss.test():', r)
            if len(r)>2 and r[2].decode().startswith('Traceback'):
                print('\033[31m"muprgrss.py" corrupted? rerun "initBoard"\033[0m')
            else:
                print('\033[32mOK\033[0m')
                res.append(p)
        else:
            print('\033[31m"muprgrss.py" not found you should run "initBoard" on port')
        ser.close()
    if debug:
        print('dbg>', 'no more ports to scan')
    return res

def initBoard(port=None, ledPin=5, npix=4, color1=None, color2=None, test=True, debug=False):
    """
    default values ledPin=5 and npix=4 for the Adafruit Neo Trinkey
    """
    if port is None:
        ports = _guessPorts()
        if len(ports)>0:
            port = ports[0]
    if port is None:
        print('please provide valid port serial')
        return False
    ser = serial.Serial(port, 115200, timeout=1)
    ser.write(b'\x04') # ctrl+d
    time.sleep(1) # give time for board to reset

    if color1 is None:
        color1 = (20,0,0)
    if color2 is None:
        color2 = (0,10,10)
    cont = ['p,n = %d, %d'%(ledPin, npix),
            'color1 = '+str(color1),
            'color2 = '+str(color2), 
            'import machine, neopixel, time',
            'np = neopixel.NeoPixel(machine.Pin(p, machine.Pin.OUT), n)',
            'def showProg(prog):',
            '   for i in range(np.n):',
            '      if i<int(prog*np.n):',
            '          np[i]=color2',
            '      elif i<prog*np.n:',
            '          x = prog*np.n-i',
            '          print(i, prog*np.n, x)',
            '          np[i]=[int(color2[j]*x+(1-x)*color1[j]) for j in range(3)]',
            '      else:',
            '          np[i]=(0,0,0)',
            '   np.write()',
            '   return',
            'def test():',
            '   x = 0',
            '   while x<=1:',
            '       showProg(x)',
            '       x+=0.01',
            '       time.sleep(0.05)',
            '   showProg(1)',
            '   time.sleep(1)',
            '   showProg(0)',
            '   return',]
    if debug:
        print('\033[37m'+'\n'.join(cont)+'\033[0m')
    code = ['import os',
            'os.remove("muprgrss.py")',
            'f = open("muprgrss.py", "w")',
            ]
    for c in cont:
        code.append("f.write('"+c+"\\n')")
    code.append('f.close()')
    code.append('')
    #print('\n'.join(code))
    
    ser.write(bytes('\r'.join(code)+'\r', 'utf8'))
    if test:
        ser.write(b'\x04') # ctrl+d
        time.sleep(0.5) # give time for board to reset
        code = ['import muprgrss', 
                'muprgrss.test()']
        ser.write(bytes('\r'.join(code)+'\r', 'utf8'))
    ser.close()
    return

_available = [p for p in _guessPorts() if _getId(p) in _trusted]
# if len(_available)>0:
#     print('available and trusted:', _available)

def progress(prog, port=None):
    global _available
    if port is None:
        # --  scan ports for progres <= 0
        if prog<=0:
            _available = [p for p in _guessPorts() if _getId(p) in _trusted]
        if len(_available)>0:
            port = _available[0]
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        ser.write(b'import muprgrss\r')
        ser.write(bytes('muprgrss.showProg(%f)\r'%prog, 'utf8'))
        #ser.readlines() # this is super slow
        ser.close()
    except:
        # -- reject port if call fails
        if port in _available:
            _available.remove(port)
    return 

def test():
    x = 0
    t = time.time()
    while x<1:
        progress(x)
        x+=1/20
        time.sleep(0.1)
    progress(1)
    print(time.time()-t)
    time.sleep(1)
    progress(0)
    return

    