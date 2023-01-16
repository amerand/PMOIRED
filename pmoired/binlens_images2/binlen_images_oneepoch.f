c      program binlen_images_oneepoch
      subroutine binlen_image(t0,u0,te,b,q,theta,rhos,t,
     +                       gcount, mxg, myg)
      implicit none
      real*8 width, radius12, radius22
      real*8 mrhomin,mgridsize,s_r_max
      real*8 b, q, alpha
      integer ncount,  DIM1,DIM2,DIM3
      real*8 sgxmax,sgymax
      parameter (DIM1=40000000)
      parameter (DIM2= 3000000)
      parameter (DIM3=  320000)

      integer sind_s(DIM2),sind_e(DIM2)
      real*4 centroidx(DIM2),centroidy(DIM2)
      real*4 centerx(DIM2),centery(DIM2)
      character*1 hbd(DIM2)
      real*4 msx(DIM1),msy(DIM1)
      character*1 cbd(DIM1)
      integer*4 boxind(DIM1)
      real*8 sgxmin,sgymin,sgrid
      real*8 rho2,magpar
      integer sgxnum,sgynum,s_ind_s,s_ind_e
      real*8 h, h_x, h_y
      real*8 scx, scy
      logical readerr

      integer gcount
cf2py intent(out) gcount
      real*4 mxg(DIM3),myg(DIM3)
cf2py intent(out) mxg
cf2py intent(out) myg

      real*8 offset_x, offset_y

      integer nparmout
      parameter (nparmout = 8)
      real*8 aoutp(nparmout)

      real*8 PI
      parameter (PI = 3.1415926535d0)

      real*8 theta
      real*8 t0, u0, te, rhos, t, tau, xs, ys

      character*7 numstr

c      open(1,file = 'parameters.txt',status='old')
c      read(1,*) aoutp
c      close(1)
c -- these are the 8 parameters in parameters.txt
c      t0    = aoutp(1)
c      u0    = aoutp(2)
c      te    = aoutp(3)
c      b     = aoutp(4)
c      q     = aoutp(5)
c      theta = aoutp(6)
c      rhos  = aoutp(7)
c      t     = aoutp(8)

ccccccccccccccccccccccccccccccccccccccccccccc
      width = 0.7d0

      radius12=(1d0+width)**2
      radius22=(1d0-width)**2

      mrhomin   = 0.1d0
      mgridsize = 0.02*0.1d0
      s_r_max   = 0.405d0
ccccccccccccccccccccccccccccccccccccccccccccc

      rho2 = rhos*rhos

      tau=(t-t0)/te

      if(b.lt.1)then
        offset_x=b*(-0.5d0+1d0/(1d0+q))
      else
        offset_x=b/2d0 -q/(1d0+q)/b
      endif

      offset_y = 0.d0

      xs= tau*cos(theta)+u0*sin(theta)  -offset_x
      ys=-tau*sin(theta)+u0*cos(theta)  -offset_y

      write(numstr, 606) t
606   format(f7.2)

c      open(98,file= 'img.'//numstr//'.txt', status='unknown')
c this function actually write the data to the file open above
      call imagemap_binary(b,q,
     +           mrhomin,width
     +           ,DIM1
     +           ,msx,msy,cbd,boxind
     +           ,ncount
     +           ,mgridsize
     +           ,scx,scy
     +           ,sgxmin,sgymin,sgxnum,sgynum
     +           ,h,h_x,h_y
     +           ,s_ind_s,s_ind_e
     +           ,readerr
     +           ,s_r_max,
     +           xs,ys,rho2,
     +           DIM3,gcount, mxg, myg)

c      close(98)

      if(readerr) then
      write(6,*) 'Map Generation Error: Program is halted'
      endif

      end
