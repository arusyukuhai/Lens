#include <stdint.h>
#define SCAN(NAME,TYPE) \
void NAME(const TYPE*w,const TYPE*s,TYPE*r,int64_t B,int64_t L,int64_t D,int trans){ \
 for(int64_t b=0;b<B;b++){ \
  if(!trans){for(int64_t i=0;i<L;i++){TYPE v=s[b*L+i]; \
   for(int64_t d=1;d<=D && d<=i;d++) v+=w[(b*L+i-d)*D+d-1]*r[b*L+i-d]; \
   r[b*L+i]=v;}} \
  else{for(int64_t i=L;i-->0;){TYPE v=s[b*L+i]; \
   for(int64_t d=1;d<=D && i+d<L;d++)v+=w[(b*L+i)*D+d-1]*r[b*L+i+d]; \
   r[b*L+i]=v;}} \
 }}
SCAN(scan64,double)
SCAN(scan32,float)
