prologues := 3;

input boxes

beginfig(1)
	circleit.vm1(btex $v_{-1}$ etex);
	circleit.v0(btex $v_0$ etex);
	circleit.v1(btex $v_1$ etex);
	circleit.v2(btex $v_2$ etex);
	circleit.v3(btex $v_3$ etex);
	circleit.v4(btex $v_4$ etex);
	circleit.v5(btex $v_5$ etex);
	circleit.px1(btex $x_1$ etex);
	circleit.px2(btex $x_2$ etex);
	circleit.pfx1x2(btex $f(x_1,x_2)$ etex);

	numeric xdist; xdist = 2cm;
	numeric ydist; ydist = 3cm;

	vm1.c-px1.c = v1.c-vm1.c = v4.c-v1.c = pfx1x2.c-v5.c = (xdist,0);
	px1.c-px2.c = vm1.c-v0.c = v4.c-v3.c = (0,ydist);
	v2.c = (xpart v1.c, ypart 1/2[v1.c,v3.c]);
	v5.c = (xpart v4.c + xdist, ypart v2.c);

	def samesize(suffix p, q) =
		p.c-p.e = q.c-q.e;
		p.c-p.s = q.c-q.s;
	enddef;

	samesize(vm1,v0);
	samesize(v0,v1);
	samesize(v0,v2);
	samesize(v0,v3);
	samesize(v0,v4);
	samesize(v0,v5);

	draw pic(px1);draw pic(px2);draw pic(pfx1x2);
	drawboxed(vm1,v0,v1,v2,v3,v4,v5);

	%   n
	% w c e
	%   s
	def sonof(suffix p, q) =
		drawarrow p.c--q.c cutbefore bpath p cutafter bpath q;
	enddef;

	sonof(pfx1x2,v5);
	sonof(v5,v4);
	sonof(v5,v3);
	sonof(v4,v1);
	sonof(v4,v2);
	sonof(v2,vm1);
	sonof(v2,v0);
	sonof(v1,vm1);
	sonof(v3,v0);
	sonof(vm1, px1);
	sonof(v0, px2);
endfig;

end;