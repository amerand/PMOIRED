      subroutine getbinp(magtot,am,xi,yi,xsi,eta,q,b,nsol)
CU    uses zroots,laguer,lenseq,dmag,conj
c      real*8 tol
      real*8 b,q
      real*8 xsi,eta
      real*8 xi(5),yi(5),am(5)
      integer*4 nsol
      real*8 magtot
      real*8 x1,m1,m2,mtot,dm,cm,mtotsq,dmsq,md,cd
      complex*16 z1,z2,z1sq,z1cu,z1fo,z1b,z2b
      complex*16 zeta,zetab,zetabsq
      complex*16 c(6),z(5),zp(5)
      logical polish
      integer*4 m
      integer*4 loop,loop1,loop2,mloop
c      complex*16 dumz,dumzb,dumzp,err,zetap,dzeta,dzetab
c      real*8 zero,detJ,derr
      complex*16 dumz,dumzb,dumzp,dzeta,dzetab
      real*8 zero,detJ
      integer soln(5)

c
c-----geometry configuration (real coordinate)
      x1 = b/2.d0
      m1 = 1.d0/(1.d0+q)
      m2 = q/(1.d0+q)
      dm = (m2-m1)/2.d0
      mtot = (m1+m2)/2.d0
      cm = x1*(dm/mtot)
      zero = 0.d0
c     
c-----geometry configuration (complex plane)
      z1 = cmplx(-x1,zero)
      z2 = -z1
      z1b = z1
      z2b = z2
      z1sq = z1*z1
      z1cu = z1sq*z1
      z1fo = z1cu*z1
c
      zeta = cmplx(xsi,eta)
      zetab =cmplx(xsi,-eta)
      zetabsq = zetab*zetab
c
      mtotsq = mtot*mtot
      dmsq = dm*dm
c
c-----calculate coefficients and solutions
      c(6) = z1sq - zetabsq
      c(5) = zeta*zetabsq-zeta*z1sq-2.d0*mtot*zetab-2.d0*dm*z1
      c(4) = 4.d0*mtot*zeta*zetab + 4.d0*dm*zetab*z1
     *     + 2.d0*zetabsq*z1sq - 2.d0*z1fo
      c(3) = 4.d0*mtotsq*zeta+4.d0*mtot*dm*z1-4.d0*dm*zeta*zetab*z1
     *     + 4.d0*dm*z1cu + 2.d0*zeta*z1fo - 2.d0*zeta*zetabsq*z1sq
      c(2) = z1fo*z1sq - z1fo*zetabsq - 4.d0*mtot*zeta*zetab*z1sq
     *     - 4.d0*dm*zetab*z1cu-4.d0*mtotsq*z1sq-4.d0*dmsq*z1sq
     *     - 8.d0*mtot*dm*zeta*z1
      c(1) = z1sq*(4.d0*dmsq*zeta + 4.d0*mtot*dm*z1 - zeta*z1fo
     *     + 4.d0*dm*zeta*zetab*z1 + 2.d0*mtot*zetab*z1sq
     *     + zeta*zetabsq*z1sq - 2.d0*dm*z1cu )
c
      m = 5
      polish = .true.
      call zroots(c,m,z,polish)
c
c-----test to see if roots are correct
      nsol = 0
      do 5 loop=1,5
         dumz = z(loop)
         dumzp = zetab + m1/(dumz-z1) + m2/(dumz-z2)
         call conj(dumzb,dumzp)
         zp(loop) = dumzb
 5    continue
      do 10 loop1=1,5
         md = 1.d1
         do 11 loop2=1,5
            cd = abs(z(loop1)-zp(loop2))
            if(cd.ge.md)go to 11
            md = cd
            mloop = loop2
 11      continue
         if(mloop.ne.loop1)then
            soln(loop1) = 0
         else
            nsol = nsol + 1
            soln(loop1) = 1
         endif
 10   continue
c
      if((nsol.ne.3).and.(nsol.ne.5))then
c         write(6,*)'Wrong nsol', nsol
      endif
c
c-----find magnifications
      do 20 loop=1,5
         if(soln(loop).eq.1)then
            dumz = z(loop)
            xi(loop) = dreal(dumz)
            yi(loop) = dimag(dumz)
            call conj(dumzb,dumz)
            dzeta = (mtot-dm)/(z1b-dumzb)**2
     *            + (mtot+dm)/(-z1b-dumzb)**2
            call conj(dzetab,dzeta)
            detJ = 1.d0 - dreal(dzeta*dzetab)
            am(loop) = abs(1.d0/detJ)
         else
            xi(loop) = 0.d0
            yi(loop) = 0.d0
            am(loop) = 0.d0
         endif
 20   continue
c
c-----calculate total magnification
      magtot = 0.d0
      do 25 loop=1,5
         magtot = magtot + abs(am(loop))
 25   continue
c
c-----return
      return

      end

*
*
      subroutine conj(zb,z)
      complex*16 z,zb
      real*8 a,b
      a = dreal(z)
      b = dimag(z)
      zb = cmplx(a,-b)
      return
      end
*
*** lenseq
*
      subroutine lenseq(zetap,z,z1,z2,m1,m2)
      real*8 m1,m2
      complex*16 zetap,z,z1,z2,z1b,z2b,zb
      call conj(z1b,z1)
      call conj(z2b,z2)
      call conj(zb,z)
      zetap = z+(m1/(z1b-zb))+(m2/(z2b-zb))
      return
      end
*
*** zroots
*
CCCCCC solve the m-th order complex polynomial
CCCCCC a(m+1)*z^m + ... + a(1) = 0
CCCCCC by Laguerre method
      subroutine zroots(a,m,roots,polish)
CU    uses laguer
      integer*4 MAXM
      real*8 EPS
      parameter (MAXM=15)
      parameter (EPS=1.d-14)
      integer*4 m
      complex*16 a(m+1),roots(m)
      complex*16 ad(MAXM)
      complex*16 x,b,c
      integer*4 i,j,jj,its
      logical polish

      do 10 j=1,m+1
         ad(j) = a(j)
 10   continue 

	zero=0.d0

      do 12 j=m,1,-1
         x = cmplx(0.,0.)
         call laguer(ad,j,x,its)
         if(abs(dimag(x)).le.2.*EPS**2*abs(dreal(x)))then
            x = cmplx(dreal(x),zero)
         endif
         roots(j) = x
         b = ad(j+1)
         do 11 jj=j,1,-1
            c = ad(jj)
            ad(jj) = b
            b = x*b + c
 11      continue
 12   continue

      do 13 j=1,m+1
         ad(j) = a(j)
 13   continue 

      if(polish)then
         do 14 j=1,m
            call laguer(ad,m,roots(j),its)
 14      continue
      endif

      do 17 j=2,m
         x = roots(j)
         do 15 i=j-1,1,-1
            if(dreal(roots(i)).le.dreal(x))go to 16
            roots(i+1) = roots(i)
 15      continue
         i = 0
 16      roots(i+1) = x
 17   continue

      return

      end
*
**** laguer
*
      subroutine laguer(ad,m,x,its)
      integer*4 MAXM,MAXIT,MR,MT
      real*8 EPSS
c      parameter (MAXM=15,MR=8,MT=10000,MAXIT=MT*MR)
      parameter (MAXM=15,MR=8,MT=100,MAXIT=MT*MR)
      parameter (EPSS=1.d-14)
c      parameter (EPSS=1.d-5)
      integer*4 m,its
      complex*16 a(m+1),x
      integer*4 i,iter,j
      real*8 abx,abp,abm,err,frac(MR),az,bz
      complex*16 ad(MAXM),dx,x1,b,d,f,g,h,gp,gm,cz
      save frac
      data frac /.5d0,.25d0,.75d0,.13d0,.38d0,.62d0,.88d0,1.d0/

      do 10 i=1,m+1
         a(i) = ad(i)
 10   continue

      do 12 iter=1,MAXIT
         its = iter
         b = a(m+1)
         err = abs(b)
         d = cmplx(0.,0.)
         f = cmplx(0.,0.)
         abx = abs(x)
         do 11 j=m,1,-1
            f = x*f + d
            d = x*d + b
            b = x*b + a(j)
            err = abs(b) + abx*err
 11      continue
         err=EPSS*err
         if(abs(b).le.err)then
            return
         else
            g = d/b
            h = g*g - f/b - f/b
            gp = g + sqrt(m*m*h+g*g-m*h-m*g*g)
            gm = g - sqrt(m*m*h+g*g-m*h-m*g*g)
            abp = abs(gp)
            abm = abs(gm)
            if(abp.lt.abm)gp=gm
            if(max(abp,abm).gt.0.)then
               dx = m/gp
            else
               az=dlog(1.+abx)
c               bz=dreal(iter)
               bz=real(iter)
               cz=cmplx(az,bz)
               dx = exp(cz)
            endif
         endif
         x1 = x - dx
         if(x.eq.x1)return
         if(mod(iter,MT).ne.0)then
            x = x1
         else
            x = x - dx*frac(iter/MT)
         endif
 12   continue

c      write(6,*) 'too many iterations in SUBROUTINE laguer'

      return

      end
