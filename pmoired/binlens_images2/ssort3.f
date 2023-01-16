c      subroutine indexsort(n,arr,brr)
      subroutine ssort3(n,xrr,yrr,crr,irr
     +           ,errflag)
      implicit none
      integer*4 n,M,NSTACK
      real*4 xrr(n),tempx,x
      real*4 yrr(n),tempy,y
      character*1 crr(n),tempc,c
      integer irr(n)
      integer*4 tempa,a
      PARAMETER (M=7,NSTACK=50)
      INTEGER i,ir,j,jstack,k,l,istack(NSTACK)
      logical errflag
      integer ihx, ihy
      real*8 h

      errflag=.false.
      
      jstack=0
      l=1
      ir=n
1     if(ir-l.lt.M)then
        do 12 j=l+1,ir
          x=xrr(j)
          y=yrr(j)
          c=crr(j)
          a=irr(j) 
          do 11 i=j-1,1,-1
            if(irr(i).le.a)goto 2
            xrr(i+1)=xrr(i)
            yrr(i+1)=yrr(i)
            crr(i+1)=crr(i)
	    irr(i+1)=irr(i)
11        continue
          i=0
2       xrr(i+1)=x
        yrr(i+1)=y
        crr(i+1)=c
	irr(i+1)=a
12      continue
        if(jstack.eq.0)return
        ir=istack(jstack)
        l=istack(jstack-1)
        jstack=jstack-2
      else
        k=(l+ir)/2
        tempx=xrr(k)
        xrr(k)=xrr(l+1)
        xrr(l+1)=tempx
        tempy=yrr(k)
        yrr(k)=yrr(l+1)
        yrr(l+1)=tempy
        tempc=crr(k)
        crr(k)=crr(l+1)
        crr(l+1)=tempc
	tempa=irr(k)
	irr(k)=irr(l+1)
	irr(l+1)=tempa
        
        if(irr(l+1).gt.irr(ir)) then
          tempx=xrr(l+1)
          xrr(l+1)=xrr(ir)
          xrr(ir)=tempx
          tempy=yrr(l+1)
          yrr(l+1)=yrr(ir)
          yrr(ir)=tempy
          tempc=crr(l+1)
          crr(l+1)=crr(ir)
          crr(ir)=tempc
	  tempa=irr(l+1)
	  irr(l+1)=irr(ir)
	  irr(ir)=tempa
        endif
        if(irr(l).gt.irr(ir)) then
          tempx=xrr(l)
          xrr(l)=xrr(ir)
          xrr(ir)=tempx
          tempy=yrr(l)
          yrr(l)=yrr(ir)
          yrr(ir)=tempy
          tempc=crr(l)
          crr(l)=crr(ir)
          crr(ir)=tempc
	  tempa=irr(l)
	  irr(l)=irr(ir)
	  irr(ir)=tempa
        endif
        if(irr(l+1).gt.irr(l)) then
          tempx=xrr(l+1)
          xrr(l+1)=xrr(l)
          xrr(l)=tempx
          tempy=yrr(l+1)
          yrr(l+1)=yrr(l)
          yrr(l)=tempy
          tempc=crr(l+1)
          crr(l+1)=crr(l)
          crr(l)=tempc
	  tempa=irr(l+1)
	  irr(l+1)=irr(l)
	  irr(l)=tempa
        endif
        i=l+1
        j=ir
        x=xrr(l)
        y=yrr(l)
        c=crr(l)
	a=irr(l)
3       continue
          i=i+1
        if(irr(i).lt.a)goto 3
4       continue
          j=j-1
        if(irr(j).gt.a)goto 4
        if(j.lt.i)goto 5
        tempx=xrr(i)
        xrr(i)=xrr(j)
        xrr(j)=tempx
        tempy=yrr(i)
        yrr(i)=yrr(j)
        yrr(j)=tempy
        tempc=crr(i)
        crr(i)=crr(j)
        crr(j)=tempc
	tempa=irr(i)
	irr(i)=irr(j)
	irr(j)=tempa

        goto 3
5       xrr(l)=xrr(j)
        xrr(j)=x
        yrr(l)=yrr(j)
        yrr(j)=y
        crr(l)=crr(j)
        crr(j)=c
	irr(l)=irr(j)
	irr(j)=a

        jstack=jstack+2
        if(jstack.gt.NSTACK) then
        write(6,*) 'NSTACK too small in sort2'
        errflag=.true.
        goto 1000
        endif
        if(ir-i+1.ge.j-l)then
          istack(jstack)=ir
          istack(jstack-1)=i
          ir=j-1
        else
          istack(jstack)=j-1
          istack(jstack-1)=l
          l=i
        endif
      endif
      goto 1
1000  return
      END
