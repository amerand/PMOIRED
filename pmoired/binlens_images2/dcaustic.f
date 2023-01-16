      subroutine gencaustics(b, q, offset_x, offset_y)
      implicit none
      real*8 b, q, offset_x, offset_y
      integer ntop
      parameter (ntop = 5000)
      real*8 x(ntop),y(ntop)
      integer i

      write(6,*) '------caustics--------------' 
      call caustics(x,y,ntop,b,q)
      write(6,*) '------caustics--------------' 

      open(9,FILE='caustics.dat',STATUS='unknown')
      do 10 i=1,ntop
      if ((abs(x(i)).LT.10).AND.(abs(y(i)).LT.10)) then
c      write(9,*) x(i)+offset_x,y(i)+offset_y
      write(9,*) (x(i)+offset_x),
     + (y(i)+offset_y)
      endif
10    continue
      call flush(9)
      close(9)

      END

      subroutine CAUSTICS(x,y,ntop,b,q)
c      implicit none
      implicit real*8 (a-h,o-z)
      real*8 x(ntop),y(ntop)
      complex*8 z(4)
      complex*8 z1,eiphi,dumz
      complex*8 c(5),conj_c,z2,zetap
      real*8 mtot,dm,m1,m2,cabs
      real*8 phi,x1,cm,pi
      integer nroots,loop,loop1
      real*8 q,b
      logical polish
      knt=0
      write(21,*)b,q
      pi=3.1415926535897932384626

      x1=b/2.0

      m2=q/(1.0+q)
      m1=1.0/(1.0+q)
      dm=(m2-m1)/2.0
      mtot=(m1+m2)/2.0
      cm=-x1*(dm/mtot)

      z2=cmplx(x1,0)
      z1=cmplx(-x1,0)

      ntop4 = ntop/4
      do loop1=1,ntop4
         phi=loop1*1./ntop4*2.0*pi

      call cmpexp_c(eiphi,-phi)

c-----calculate coefficients

      c(1)=-2.0*mtot*z1**2 + eiphi*z1**4
      c(2)=4.0*z1*dm
      c(3)=- 2.0*mtot-2.0*eiphi*z1**2
      c(4)=0.0
      c(5)=eiphi

      nroots=4

      polish=.false.

      call zroots_c(c,nroots,z,polish)

      do loop=1,4

      dumz=z(loop)
      call lenseq_c(zetap,dumz,z1,z2,m1,m2)

******               Caustic   *********   Critical Curve  
*********************************************************
******          x_pos      ,y_pos       ,x_pos     ,y_pos
*********************************************************
c      write(20+loop,*)real(zetap),aimag(zetap),real(dumz),aimag(dumz)
cc--replaced      write(20+loop,*)real(zetap),aimag(zetap)
      kpos = ntop4*(loop-1) + loop1
      x(kpos) = real(zetap)
      y(kpos) = aimag(zetap)
c      write(20+loop,*)zetap
c      write(20+loop,*)x(kpos),y(kpos)
c      write(25,*)x(kpos),y(kpos)
      knt = knt + 1
      enddo

      enddo
      write(6,*)knt
      end




      SUBROUTINE CMPEXP_C(cexp,theta)
      complex*8 cexp
      real*8 theta


      cexp=cmplx(cos(theta),sin(theta))

      return
      end

      SUBROUTINE zroots_c(a,m,roots,polish)
      INTEGER m,MAXM
      REAL*8 EPS
      COMPLEX*8 a(m+1),roots(m)
      LOGICAL polish
      PARAMETER (EPS=1.e-7,MAXM=501)
CU    USES laguer
      INTEGER i,j,jj,its
      COMPLEX*8 ad(MAXM),x,b,c
      do 11 j=1,m+1
        ad(j)=a(j)
11    continue
      do 13 j=m,1,-1
        x=cmplx(0.0,0.0)
        call laguer_c(ad,j,x,its)
        if(abs(aimag(x)).le.2.*EPS**2*abs(real(x))) x=cmplx(real(x),0.)
        roots(j)=x
        b=ad(j+1)
        do 12 jj=j,1,-1
          c=ad(jj)
          ad(jj)=b
          b=x*b+c
12      continue
13    continue


      if (polish) then
        do 14 j=1,m
          call laguer_c(a,m,roots(j),its)
14      continue
      endif
      do 16 j=2,m
        x=roots(j)
        do 15 i=j-1,1,-1
          if(real(roots(i)).le.real(x))goto 10
          roots(i+1)=roots(i)
15      continue
        i=0
10      roots(i+1)=x
16    continue
      return
      END


      SUBROUTINE laguer_c(a,m,x,its)
      INTEGER m,its,MAXIT,MR,MT
      REAL*8 EPSS
      COMPLEX*8 a(m+1),x
      PARAMETER (EPSS=2.e-7,MR=8,MT=10,MAXIT=MT*MR)
      INTEGER iter,j
      REAL*8 abx,abp,abm,err,frac(MR)
      COMPLEX*8 dx,x1,b,d,f,g,h,sq,gp,gm,g2
      SAVE frac
      DATA frac /.5,.25,.75,.13,.38,.62,.88,1./
      do 12 iter=1,MAXIT
        its=iter 
        b=a(m+1)
        err=abs(b)
        d=cmplx(0.,0.)
        f=cmplx(0.,0.)
        abx=abs(x)
        do 11 j=m,1,-1
          f=x*f+d
          d=x*d+b
          b=x*b+a(j)
          err=abs(b)+abx*err
11      continue
        err=EPSS*err
        if(abs(b).le.err) then
          return
        else

          g=d/b
          g2=g*g
          h=g2-2.*f/b
          sq=sqrt((m-1)*(m*h-g2))
          gp=g+sq
          gm=g-sq
          abp=abs(gp)
          abm=abs(gm)
          if(abp.lt.abm) gp=gm
          if (max(abp,abm).gt.0.) then
            dx=m/gp
          else
            dx=exp(cmplx(log(1.+abx),real(iter)))
          endif
        endif
        x1=x-dx
        if(x.eq.x1)return
        if (mod(iter,MT).ne.0) then
          x=x1
        else
          x=x-dx*frac(iter/MT)
        endif
12    continue
      write(6,*) 'too many iterations in laguer'
      return
      END


      COMPLEX FUNCTION CONJ_C(Z)

      complex*8 z
      real*8 a,b

      a=real(z)
      b=aimag(z)

      conj_c=cmplx(a,-b)

      return
      end




      SUBROUTINE LENSEQ_C(zetap,z,z1,z2,m1,m2)

      complex*8 z1,z2
      real*8 m1,m2
      complex*8 z,zetap,conj_c

      zetap = z+(m1/(conj_c(z1)-conj_c(z)))
     + +(m2/(conj_c(z2)-conj_c(z)))
      return
      end

