c       implicit real*8 (a-h,o-z)
c       h = 1
c       xmin =0
c       ymin = 0
c       do 20 ii=1,100
c          x = ii/100.*15
c          do 20 jj =1,100
c             y = jj/100.*15
c       call hexloc(ix,jy,x,y,xmin,ymin,h)
c       ixp = mod(ix+4,2)
c       jyp = mod(jy+4,2)
c       icol = 2*ixp + jyp + 1
c       write(17,21)icol,x,y
c  21   format(i1,2f8.4)
c  20   continue
c       stop
c       end
c
       subroutine hexloc(ix,jy,x,y,xmin,ymin,h, sqrt3)
       implicit real*8 (a-h,o-z)
cccccc       sqrt3 = sqrt(3.)
       xrel = (x - xmin)/h
       yrel = (y - ymin)/h
       ly = nint((yrel-0.25)/1.5)
       yoff = yrel - 1.5d0*ly - 0.5
       xadd = 0
       if(2*(ly/2)-ly.ne.0)xadd = 0.5
  3    format(i3,4f9.5)
       if(yoff.le.0.)then
          ix = nint(xrel/sqrt3 + xadd)
          jy = ly
          return
       endif
       xrelp = xrel/sqrt3 + xadd + yoff
       kx = nint(xrelp)
       xoff = xrelp - kx + 0.5
       if(xoff.gt.2*yoff)then
          ix = kx
          jy = ly
       else
          if(xadd.eq.0)then
             ix = kx
             jy = ly + 1
          else
             ix = kx - 1
             jy = ly + 1
          endif
       endif
       return
       end
