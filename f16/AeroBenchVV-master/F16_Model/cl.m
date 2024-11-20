function cll=cl(alpha,beta)
a=[0 0 0 0 0 0 0 0 0 0 0 0
-.001 -.004 -.008 -.012 -.016 -.022 -.022 -.021 -.015 -.008 -.013 -.015
-.003 -.009 -.017 -.024 -.030 -.041 -.045 -.040 -.016 -.002 -.010 -.019
-.001 -.010 -.020 -.030 -.039 -.054 -.057 -.054 -.023 -.006 -.014 -.027
.000 -.010 -.022 -.034 -.047 -.060 -.069 -.067 -.033 -.036 -.035 -.035
.007 -.010 -.023 -.034 -.049 -.063 -.081 -.079 -.060 -.058 -.062 -.059
.009 -.011 -.023 -.037 -.050 -.068 -.089 -.088 -.091 -.076 -.077 -.076]';
s=.2*alpha;
k=fix(s);
if(k<=-2),k=-1;end
if(k>=9),k=8;end
da=s-k;
l=k+fix(1.1*sign(da));
s=.2*abs(beta);
m=fix(s);
if(m==0),m=1;end
if(m>=6),m=5;end
db=s-m;
n=m+fix(1.1*sign(db));
l=l+3;
k=k+3;
m=m+1;
n=n+1;
t=a(k,m);
u=a(k,n);
v=t+abs(da)*(a(l,m)-t);
w=u+abs(da)*(a(l,n)-u);
dum=v+(w-v)*abs(db);
cll=dum*sign(beta);