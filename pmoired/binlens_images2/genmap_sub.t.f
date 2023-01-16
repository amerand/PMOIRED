      subroutine imagemap_binary(b, q,
     +            rhomin,width
     +           ,DIM1
     +           ,msx,msy,cbd,boxind
     +           ,ncount
     +           ,mgridsize
     +           ,scx,scy
     +           ,sgxmin,sgymin,sgxnum,sgynum
     +           ,h,h_x,h_y
     +           ,msindexs,msindexe
     +           ,errflag
     +           ,s_r_max,
     +           xs,ys,rhos2,
     +           DIM3, gcount, mxg, myg)
      implicit none

      logical errflag
      integer*4 DIM1
      integer DIM3

c     Dimension of the largest array for the points on the source plane
      real*4 msx(DIM1),msy(DIM1)
      real*4 mxg(DIM3),myg(DIM3)

      character*1 cbd(DIM1)
      integer*4 boxind(DIM1)

c     X,Y coordinates and whether on boundary
c     'F' -> not on boundary
c     'T' -> on boundary
ccccccccccccccccc      real*8 b,q,m1,m2,lgrhomin(10),u1,u2,bdiv2,rhomin
      real*8 u1,u2,bdiv2
      real*8 rhomin
      real*8 b, q
      real*8 offset_x, offset_y
c     Offset needed to do the coordinate transformation from
c     the center of the star and planet to the center of magnification
      real*8 scx,scy,radius1,radius2
c     Center of the source plane
c     radius1 and radius2 are the boundary on the image plane
      real*8 xorigin,yorigin,width,dist,dm1,dm2
      real*8 mgridsize
      real*8 PI
      real*8 xmin,xmax,ymin,ymax,xgmin,ygmin,xg,yg
      real*8 sx,sy,sxmin,sxmax,symin,symax
      real*8 sgrid,sgxmin,sgymin
      integer*4 sgxnum,sgynum,gx,gy
      parameter (PI=3.141592653589793238462643d0)
      real*8 RADIAN
      parameter(RADIAN=1.8d2/PI)
      integer*4 xnum,ynum,i,j,ncount, ihx, ihy
      integer*4 msindexs,msindexe
      real*8 s_r_max, s_r_max2
      real*8 h, h_x, h_y, sdist
      real*8 xtemp, ytemp
      integer mpair, npair, ipair
      parameter (mpair = 6)
      integer ipair_s(mpair), ipair_e(mpair)
      character*1 ipair_b(mpair)
      real*8 ydist
      real*8 xc1_min, xc1_max
      integer ic1_min, ic1_max
      real*8 xc2_min, xc2_max
      integer ic2_min, ic2_max
      real*8 xm1_min, xm1_max
      integer im1_min, im1_max
      real*8 xm2_min, xm2_max
      integer im2_min, im2_max
      real*8 radius1_m, radius2_m
      real time1, time2, etime, time(2)

      integer mcount
      integer*4 gcount

      real*8 sqrt3
      real*8 xs, ys, rhos2, rdist2

      bdiv2=b/2d0
      u1=1d0+q
      u2=1d0+1d0/q

      sqrt3 = sqrt(3.d0)

      time1 = etime(time)
      time2 = etime(time)

      h=rhomin*0.03d0
c      h=rhomin*0.15d0

      h_x = sqrt3*h
      h_y = 1.5d0 * h

c      s_r_max  = width
      s_r_max2 = s_r_max*s_r_max

      errflag=.false.

      do i=1,DIM1
        msx(i)=0.0
        msy(i)=0.0
        cbd(i)='F'
        boxind(i) = 0
      enddo

      do i=1,DIM3
        mxg(i)=0.0
        myg(i)=0.0
      enddo

ccccccccccc      if(b.lt.1)then
c          offset=b*(-0.5d0+1d0/(1d0+q))
cccc          offset_x = -(da*0.5d0*(1.d0-ma-mb)/(1.d0+ma+mb)
cccc     + - db*mb*dcos(phi)/(1.d0+ma+mb))
cccc          offset_y = -(mb*db*dsin(phi)/(1.d0+ma+mb))
      if(b.lt.1)then
          offset_x=b*(-0.5d0+1d0/(1d0+q))
      else
          offset_x=b/2d0 -q/(1d0+q)/b
      endif

          offset_y = 0.d0
c    Center of Mass Convention!!!
cccccccccccc      else
c          offset=b/2d0 -q/(1d0+q)/b
cccccccccccc      endif
      scx=0d0-offset_x
      scy=0d0-offset_y
      xorigin=scx
      yorigin=scy
      xmin=xorigin-(1d0+width)
      xmax=xorigin+(1d0+width)
      ymin=scy-(1d0+width)
      ymax=scy+(1d0+width)
ch     refine later, maybe not offset!
chhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh
      sxmin  = scx - s_r_max - h_x
      sxmax  = scx + s_r_max + h_x
      symin  = scy - s_r_max - h_y
      symax  = scy + s_r_max + h_y

      call buildgrid2(sxmin,sxmax,scx,h_x,sgxmin,sgxnum)
      call buildgrid2(symin,symax,scy,h_y,sgymin,sgynum)

      radius1=1d0+width
      radius2=1d0-width
      radius1_m = radius1 - mgridsize
      radius2_m = radius2 + mgridsize
      call buildgrid3(xmin,xmax,xorigin,mgridsize,xgmin,xnum)
      call buildgrid3(ymin,ymax,yorigin,mgridsize,ygmin,ynum)

      ncount=0
      mcount=0
      gcount=0

      do 100 i=1,ynum
      yg=ygmin+mgridsize*(i-1)

      ydist = abs(yg - scy)

      if(ydist.gt.radius1) goto 100

      xc1_min = xorigin - sqrt(radius1**2 - ydist**2)
      xc1_max = 2.d0*xorigin - xc1_min
      call more_grid(xc1_min, xorigin, mgridsize, ic1_min)
      call less_grid(xc1_max, xorigin, mgridsize, ic1_max)
c---------------------------------------------------------------
      if(ydist.le.radius1_m) then
      xm1_min    = xorigin - sqrt(radius1_m**2 - ydist**2)
      xm1_max    = 2.d0*xorigin - xm1_min
      call less_grid(xm1_min, xorigin, mgridsize, im1_min)
      call more_grid(xm1_max, xorigin, mgridsize, im1_max)
      endif

      if(ydist.le.radius2_m) then
      xm2_min = xorigin - sqrt(radius2_m**2 - ydist**2)
      xm2_max = 2.d0*xorigin - xm2_min
      call more_grid(xm2_min, xorigin, mgridsize, im2_min)
      call less_grid(xm2_max, xorigin, mgridsize, im2_max)
      endif

      if(ydist.le.radius2) then
      xc2_min = xorigin - sqrt(radius2**2 - ydist**2)
      xc2_max = 2.d0*xorigin - xc2_min
      call less_grid(xc2_min, xorigin, mgridsize, ic2_min)
      call more_grid(xc2_max, xorigin, mgridsize, ic2_max)
      endif
c---------------------------------------------------------------

      if(ydist.gt.radius1_m) then
        npair = 1
        ipair_s(1) = ic1_min
        ipair_e(1) = ic1_max
        ipair_b(1) = 'T'
      else if(ydist.gt.radius2_m) then
        npair = 2
        ipair_s(1) = ic1_min
        ipair_e(1) = im1_min
        ipair_b(1) = 'T'
        ipair_s(2) = im1_max
        ipair_e(2) = ic1_max
        ipair_b(2) = 'T'

      if ((im1_max - 1).ge.(im1_min+1)) then
        npair = 3
        ipair_s(3) = im1_min + 1
        ipair_e(3) = im1_max - 1
        ipair_b(3) = 'F'
      endif
      else if(ydist.gt.radius2) then
        npair = 5
        ipair_s(1) = ic1_min
        ipair_e(1) = im1_min
        ipair_b(1) = 'T'
        ipair_s(2) = im1_min + 1
        ipair_e(2) = im2_min - 1
        ipair_b(2) = 'F'
        ipair_s(3) = im2_min
        ipair_e(3) = im2_max
        ipair_b(3) = 'T'
        ipair_s(4) = im2_max + 1
        ipair_e(4) = im1_max - 1
        ipair_b(4) = 'F'
        ipair_s(5) = im1_max
        ipair_e(5) = ic1_max
        ipair_b(5) = 'T'
      else
        npair = 6
        ipair_s(1) = ic1_min
        ipair_e(1) = im1_min
        ipair_b(1) = 'T'
        ipair_s(2) = im1_min + 1
        ipair_e(2) = im2_min - 1
        ipair_b(2) = 'F'
        ipair_s(3) = im2_min
        ipair_e(3) = ic2_min
        ipair_b(3) = 'T'
        ipair_s(4) = im2_max + 1
        ipair_e(4) = im1_max - 1
        ipair_b(4) = 'F'
        ipair_s(5) = im1_max
        ipair_e(5) = ic1_max
        ipair_b(5) = 'T'

	  if(ic2_max.eq.ic2_min) then
		 if((ic2_max+1).le.im2_max) then
                 ipair_s(6) = ic2_max+1
                 ipair_e(6) = im2_max
                 ipair_b(6) = 'T'
                 else
                 npair = 5
                 endif
	  else
                 ipair_s(6) = ic2_max
                 ipair_e(6) = im2_max
                 ipair_b(6) = 'T'
	  endif

      endif

c      if(i.eq.1) then
c        write (*,*) 'npair:', npair
c        do 99 j=1,npair
c          write (*,*) 'ipair_s:', j, ipair_s(j)
c          write (*,*) 'ipair_e:', j, ipair_e(j)
c99        write (*,*) 'ipair_b:', j, ipair_b(j)
c      endif


      do 201 ipair = 1,npair
         do 202 j=ipair_s(ipair), ipair_e(ipair)

      xg=xorigin+mgridsize*(j-1)

      call inverseshoot2(sx,sy,xg,yg,bdiv2,u1,u2)
c      call
c     + inverseshoot3(sx,sy,xg,yg, x1, x2, x3, y3, q1, q2, q3)
      sdist = (sx - scx)**2 + (sy - scy)**2
      if(sdist.gt.s_r_max2) goto 300
      rdist2 = (sx - xs)**2 + (sy - ys)**2
      if(rdist2.le.rhos2) then
c        write(98,*) xg, yg
        gcount=gcount+1
        mxg(gcount) = xg
        myg(gcount) = yg
      endif
      ncount=ncount+1

      if(gcount.gt.DIM3) then
        write(6,*) 'not enough space to store xg, yg'
        call flush(16)
        errflag=.true.
        goto 1000
      endif


      if(ncount.gt.DIM1) then
        write(*,*) 'Dimension Definition Error'
        call flush(16)
        errflag=.true.
        goto 1000
      endif

      cbd(ncount) = ipair_b(ipair)

      msx(ncount)=sx
      msy(ncount)=sy
      call hexloc(ihx,ihy,sx,sy,sgxmin,sgymin,h,sqrt3)
      boxind(ncount) = ihx + 1 + ihy*sgxnum

300   continue
202   continue
201   continue
100   continue

c added by Antoine, as everything below seems useless for our intent
      goto 1000

c      call ssort(ncount,msx,msy,cbd
c     +           ,sgxmin,sgymin,h,sgxnum
c     +           ,errflag)
      call ssort3(ncount,msx,msy,cbd,boxind,errflag)

      if(errflag) goto 1000

c      xtemp = msx(1)
c      ytemp = msy(1)
c      call hexloc(ihx,ihy,xtemp,ytemp,sgxmin,sgymin,h,sqrt3)
c      msindexs= ihx + 1 + ihy*sgxnum
      msindexs= boxind(1)
c      xtemp = msx(ncount)
c      ytemp = msy(ncount)
c      call hexloc(ihx,ihy,xtemp,ytemp,sgxmin,sgymin,h, sqrt3)
c      msindexe= ihx + 1 + ihy*sgxnum
      msindexe=boxind(ncount)

1000  return
      end

      subroutine buildgrid3(mini,maxi,origin,gridsize,
     +           gridmin,gridnum)
      implicit none
      real*8 mini,maxi,origin,gridsize,gridmin
      integer gridnum,n1,n2

      n1=nint((mini-origin)/gridsize+0.4999999999d0)
      n2=nint((maxi-origin)/gridsize-0.4999999999d0)

      gridmin=n1*gridsize+origin
      gridnum=n2-n1+1

      return
      end

      subroutine buildgrid2(mini,maxi,origin,gridsize,
     +           gridmin,gridnum)
      implicit none
      real*8 mini,maxi,origin,gridsize,gridmin
      integer gridnum,n1,n2

      n1=nint((mini-origin)/gridsize-0.4999999999d0)
      n2=nint((maxi-origin)/gridsize+0.4999999999d0)

c      gridmin=n1*gridsize+origin
      gridmin=n1*gridsize+origin + gridsize*0.5d0
ch    changed!!!
      gridnum=n2-n1+1
ch    changed!!!

      return
      end

      subroutine less_grid(x, origin, gridsize, ind)
      implicit none
      real*8 x, origin, gridsize
      integer ind
      ind = nint((x-origin)/gridsize - 0.4999999999d0)
      return
      end

      subroutine more_grid(x, origin, gridsize, ind)
      implicit none
      real*8 x, origin, gridsize
      integer ind
      ind = nint((x-origin)/gridsize + 0.4999999999d0)
      return
      end

      subroutine inverseshoot2(xst,yst,xit,yit,ddiv2,u1,u2)
cccccccccccccccccccccccccccccccccccccccccccccccccccc
c      u1=m1^(-1)
c      u2=m2^(-1)
       real*8 xst,yst,xit,yit,ddiv2,u1,u2
       real*8 xipd,ximd,xipd2,ximd2,yi2
       real*8 d1,d2,d3,d4
c      m1=1.0/(1.0+q)
c      m2=q/(1.0+q)
       xipd=xit+ddiv2
       ximd=xit-ddiv2
       xipd2=xipd*xipd
       ximd2=ximd*ximd
       yi2=yit*yit
       d1=xipd2+yi2
       d2=ximd2+yi2
       d3=d1*u1
       d4=d2*u2
       xst=xit-xipd/d3-ximd/d4
       yst=yit-yit/d3-yit/d4
c      xst=xit-m1*(xit+d/2.0)/((xit+d/2.)**2+yit**2)-
c     &m2*(xit-d/2.0)/((xit-d/2.)**2+yit**2)
c      yst=yit-m1*yit/((xit+d/2.)**2+yit**2)-
c     &m2*yit/((xit-d/2.)**2+yit**2)
      return
      end

      subroutine inverseshoot3
     + (xst, yst, xit, yit, x1, x2, x3, y3, q1, q2, q3)
      implicit none
      real*8 xst, yst, xit, yit, x1, x2, x3, y3, q1, q2, q3
      real*8 r1, r2, r3
      real*8 q1r1, q2r2, q3r3
       r1 = (xit-x1)**2+yit**2
       r2 = (xit-x2)**2+yit**2
       r3 = (xit-x3)**2+(yit-y3)**2
       q1r1 = q1/r1
       q2r2 = q2/r2
       q3r3 = q3/r3
       xst = xit - q1r1*(xit-x1) -
     + q2r2*(xit-x2) -
     + q3r3*(xit-x3)
       yst = yit - q1r1*yit -
     + q2r2*yit -
     + q3r3*(yit-y3)
       return
       end
