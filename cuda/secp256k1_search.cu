typedef unsigned char uint8_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef int int32_t;

namespace {
using Limb = uint32_t;
using Wide = uint64_t;
struct U256 { Limb v[8]; };
struct Point { U256 x; U256 y; U256 z; bool infinity; };

__device__ __constant__ U256 FIELD_P = {{
    0xfffffc2fU, 0xfffffffeU, 0xffffffffU, 0xffffffffU,
    0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU
}};
__device__ __constant__ U256 FIELD_P_MINUS_2 = {{
    0xfffffc2dU, 0xfffffffeU, 0xffffffffU, 0xffffffffU,
    0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU
}};
__device__ __constant__ U256 GROUP_N = {{
    0xd0364141U, 0xbfd25e8cU, 0xaf48a03bU, 0xbaaedce6U,
    0xffffffffU, 0xffffffffU, 0xffffffffU, 0xffffffffU
}};

__device__ __forceinline__ U256 zero_u256() { U256 r = {{0,0,0,0,0,0,0,0}}; return r; }
__device__ __forceinline__ U256 one_u256() { U256 r = {{1,0,0,0,0,0,0,0}}; return r; }
__device__ __forceinline__ bool is_zero(const U256& a) {
    Limb x = 0;
    #pragma unroll
    for (int i=0;i<8;i++) x |= a.v[i];
    return x == 0;
}
__device__ __forceinline__ bool ge(const U256& a, const U256& b) {
    for (int i=7;i>=0;i--) if (a.v[i] != b.v[i]) return a.v[i] > b.v[i];
    return true;
}
__device__ __forceinline__ U256 raw_sub(const U256& a,const U256& b) {
    U256 r; Wide borrow=0;
    #pragma unroll
    for(int i=0;i<8;i++){ Wide bi=(Wide)b.v[i]+borrow; r.v[i]=(Limb)((Wide)a.v[i]-bi); borrow=(Wide)a.v[i]<bi; }
    return r;
}
__device__ __forceinline__ U256 fadd(const U256& a,const U256& b) {
    U256 r; Wide carry=0;
    #pragma unroll
    for(int i=0;i<8;i++){ Wide s=(Wide)a.v[i]+b.v[i]+carry; r.v[i]=(Limb)s; carry=s>>32; }
    if(carry || ge(r,FIELD_P)) r=raw_sub(r,FIELD_P);
    return r;
}
__device__ __forceinline__ U256 fsub(const U256& a,const U256& b) {
    if(ge(a,b)) return raw_sub(a,b);
    return fadd(raw_sub(FIELD_P,b),a);
}
__device__ void add_product_limb(uint32_t* limbs, int position, uint64_t value) {
    while (value != 0 && position < 20) {
        uint64_t sum = (uint64_t)limbs[position] + (value & 0xffffffffULL);
        limbs[position] = (uint32_t)sum;
        value = (value >> 32) + (sum >> 32);
        ++position;
    }
}

// Schoolbook 8x32 multiplication followed by pseudo-Mersenne folding using
// 2^256 = 2^32 + 977 (mod p). This replaces the 256-round add-and-double mul.
__device__ __noinline__ U256 fmul(const U256& a,const U256& b) {
    uint32_t product[20] = {0};
    for (int i=0;i<8;i++) {
        uint64_t carry=0;
        for (int j=0;j<8;j++) {
            uint64_t sum=(uint64_t)a.v[i]*b.v[j]+product[i+j]+carry;
            product[i+j]=(uint32_t)sum;
            carry=sum>>32;
        }
        int k=i+8;
        while (carry != 0) {
            uint64_t sum=(uint64_t)product[k]+carry;
            product[k]=(uint32_t)sum;
            carry=sum>>32;
            ++k;
        }
    }
    for (int pass=0;pass<4;pass++) {
        for (int k=19;k>=8;k--) {
            uint32_t high=product[k];
            if (high != 0) {
                product[k]=0;
                add_product_limb(product,k-8,(uint64_t)high*977ULL);
                add_product_limb(product,k-7,high);
            }
    }
    }
    U256 r;
    #pragma unroll
    for(int i=0;i<8;i++) r.v[i]=product[i];
    while(ge(r,FIELD_P)) r=raw_sub(r,FIELD_P);
    return r;
}
__device__ __forceinline__ bool is_one(const U256& a) {
    Limb rest=0;
    #pragma unroll
    for(int i=1;i<8;i++) rest|=a.v[i];
    return a.v[0]==1 && rest==0;
}
__device__ __forceinline__ bool equal_u256(const U256& a,const U256& b) {
    Limb diff=0;
    #pragma unroll
    for(int i=0;i<8;i++) diff|=a.v[i]^b.v[i];
    return diff==0;
}
__device__ __forceinline__ U256 half_mod(const U256& x) {
    U256 r; Wide carry=0;
    if(x.v[0]&1U) {
        #pragma unroll
        for(int i=0;i<8;i++) { Wide sum=(Wide)x.v[i]+FIELD_P.v[i]+carry; r.v[i]=(Limb)sum; carry=sum>>32; }
    } else {
        r=x;
    }
    #pragma unroll
    for(int i=0;i<7;i++) r.v[i]=(r.v[i]>>1)|(r.v[i+1]<<31);
    r.v[7]=(r.v[7]>>1)|((Limb)carry<<31);
    return r;
}
__device__ __noinline__ U256 finv(const U256& a) {
    // Binary extended Euclid avoids hundreds of full modular multiplications.
    if(is_zero(a)) return zero_u256();
    U256 u=a, v=FIELD_P, x1=one_u256(), x2=zero_u256();
    while(!is_one(u) && !is_one(v)) {
        while((u.v[0]&1U)==0) { u=half_mod(u); x1=half_mod(x1); }
        while((v.v[0]&1U)==0) { v=half_mod(v); x2=half_mod(x2); }
        if(ge(u,v)) { u=raw_sub(u,v); x1=fsub(x1,x2); }
        else { v=raw_sub(v,u); x2=fsub(x2,x1); }
    }
    return is_one(u)?x1:x2;
}
__device__ __forceinline__ U256 shfl_up_u256(const U256& value, int delta) {
    U256 out;
    #pragma unroll
    for(int i=0;i<8;i++) out.v[i]=__shfl_up_sync(0xffffffffU,value.v[i],delta);
    return out;
}
__device__ __forceinline__ U256 shfl_down_u256(const U256& value, int delta) {
    U256 out;
    #pragma unroll
    for(int i=0;i<8;i++) out.v[i]=__shfl_down_sync(0xffffffffU,value.v[i],delta);
    return out;
}
__device__ __forceinline__ U256 shfl_u256(const U256& value, int source_lane) {
    U256 out;
    #pragma unroll
    for(int i=0;i<8;i++) out.v[i]=__shfl_sync(0xffffffffU,value.v[i],source_lane);
    return out;
}
// One field inversion per warp: each lane receives the inverse of its input.
// Inputs must be nonzero; inactive lanes contribute one.
__device__ __noinline__ U256 warp_batch_inverse(const U256& value) {
    const int lane=threadIdx.x&31;
    U256 prefix=value;
    #pragma unroll
    for(int offset=1;offset<32;offset<<=1) {
        U256 left=shfl_up_u256(prefix,offset);
        if(lane>=offset) prefix=fmul(prefix,left);
    }
    U256 suffix=value;
    #pragma unroll
    for(int offset=1;offset<32;offset<<=1) {
        U256 right=shfl_down_u256(suffix,offset);
        if(lane+offset<32) suffix=fmul(suffix,right);
    }
    U256 total=shfl_u256(prefix,31);
    U256 total_inverse=zero_u256();
    if(lane==31) total_inverse=finv(total);
    total_inverse=shfl_u256(total_inverse,31);
    U256 prefix_before=shfl_up_u256(prefix,1);
    U256 suffix_after=shfl_down_u256(suffix,1);
    if(lane==0) prefix_before=one_u256();
    if(lane==31) suffix_after=one_u256();
    return fmul(fmul(total_inverse,prefix_before),suffix_after);
}
__device__ U256 scalar_sub(const U256& a,const U256& b) {
    if(ge(a,b)) return raw_sub(a,b);
    U256 r; Wide carry=0;
    U256 x=raw_sub(GROUP_N,b);
    for(int i=0;i<8;i++){ Wide s=(Wide)x.v[i]+a.v[i]+carry; r.v[i]=(Limb)s; carry=s>>32; }
    if(carry || ge(r,GROUP_N)) r=raw_sub(r,GROUP_N);
    return r;
}
__device__ U256 scalar_add(const U256& a,const U256& b) {
    U256 r; Wide carry=0;
    for(int i=0;i<8;i++){ Wide s=(Wide)a.v[i]+b.v[i]+carry; r.v[i]=(Limb)s; carry=s>>32; }
    if(carry || ge(r,GROUP_N)) r=raw_sub(r,GROUP_N);
    return r;
}
__device__ U256 scalar_mul_u64(const U256& a,uint64_t b) {
    uint32_t factor[2]={(uint32_t)b,(uint32_t)(b>>32)};
    uint32_t product[10]={0};
    for(int i=0;i<8;i++) {
        uint64_t carry=0;
        for(int j=0;j<2;j++) {
            uint64_t sum=(uint64_t)a.v[i]*factor[j]+product[i+j]+carry;
            product[i+j]=(uint32_t)sum;
            carry=sum>>32;
        }
        int k=i+2;
        while(carry && k<10) {
            uint64_t sum=(uint64_t)product[k]+carry;
            product[k]=(uint32_t)sum;
            carry=sum>>32;
            ++k;
        }
    }
    U256 result;
    #pragma unroll
    for(int i=0;i<8;i++) result.v[i]=product[i];
    return result;
}
__device__ U256 load_be256(const uint8_t* b) {
    U256 r;
    #pragma unroll
    for(int i=0;i<8;i++){ int o=32-(i+1)*4; r.v[i]=((uint32_t)b[o]<<24)|((uint32_t)b[o+1]<<16)|((uint32_t)b[o+2]<<8)|b[o+3]; }
    return r;
}
__device__ void store_be256(const U256& value, uint8_t* out) {
    #pragma unroll
    for(int i=0;i<8;i++) { uint32_t w=value.v[7-i]; out[i*4]=(uint8_t)(w>>24); out[i*4+1]=(uint8_t)(w>>16); out[i*4+2]=(uint8_t)(w>>8); out[i*4+3]=(uint8_t)w; }
}
__device__ Point infinity_point(){ Point p; p.x=zero_u256();p.y=zero_u256();p.z=zero_u256();p.infinity=true;return p; }
__device__ __noinline__ Point jacobian_double(const Point& p) {
    if(p.infinity || is_zero(p.y)) return infinity_point();
    U256 yy=fmul(p.y,p.y), yyyy=fmul(yy,yy), xx=fmul(p.x,p.x);
    U256 s=fmul(p.x,yy); s=fadd(fadd(s,s),fadd(s,s));
    U256 m=fadd(xx,fadd(xx,xx));
    U256 x3=fsub(fmul(m,m),fadd(s,s));
    U256 eight=fadd(fadd(yyyy,yyyy),fadd(yyyy,yyyy)); eight=fadd(eight,eight);
    U256 y3=fsub(fmul(m,fsub(s,x3)),eight);
    U256 z3=fadd(fmul(p.y,p.z),fmul(p.y,p.z));
    Point r={x3,y3,z3,false}; return r;
}
__device__ __noinline__ Point jacobian_add_affine(const Point& p,const U256& qx,const U256& qy) {
    if(p.infinity) return Point{qx,qy,one_u256(),false};
    U256 z2=fmul(p.z,p.z), z3=fmul(z2,p.z);
    U256 h=fsub(fmul(qx,z2),p.x), r=fsub(fmul(qy,z3),p.y);
    if(is_zero(h)){ if(is_zero(r)) return jacobian_double(p); return infinity_point(); }
    U256 hh=fmul(h,h), hhh=fmul(hh,h), v=fmul(p.x,hh);
    U256 x3=fsub(fsub(fmul(r,r),hhh),fadd(v,v));
    U256 y3=fsub(fmul(r,fsub(v,x3)),fmul(p.y,hhh));
    U256 z_out=fmul(p.z,h);
    return Point{x3,y3,z_out,false};
}
__device__ __forceinline__ Point affine_add_points(const Point& a,const Point& b) {
    if(a.infinity) return b;
    if(b.infinity) return a;
    if(equal_u256(a.x,b.x)) {
        if(!equal_u256(a.y,b.y) || is_zero(a.y)) return infinity_point();
        U256 x2=fmul(a.x,a.x);
        U256 slope=fmul(fadd(x2,fadd(x2,x2)),finv(fadd(a.y,a.y)));
        U256 x3=fsub(fmul(slope,slope),fadd(a.x,a.x));
        U256 y3=fsub(fmul(slope,fsub(a.x,x3)),a.y);
        return Point{x3,y3,one_u256(),false};
    }
    U256 slope=fmul(fsub(b.y,a.y),finv(fsub(b.x,a.x)));
    U256 x3=fsub(fsub(fmul(slope,slope),a.x),b.x);
    U256 y3=fsub(fmul(slope,fsub(a.x,x3)),a.y);
    return Point{x3,y3,one_u256(),false};
}
__device__ void serialize_affine(const U256& x,const U256& y,uint8_t* out) {
    out[0]=(uint8_t)(2U|(y.v[0]&1U));
    #pragma unroll
    for(int i=0;i<8;i++){ uint32_t w=x.v[7-i]; out[1+i*4]=(uint8_t)(w>>24);out[2+i*4]=(uint8_t)(w>>16);out[3+i*4]=(uint8_t)(w>>8);out[4+i*4]=(uint8_t)w; }
}
__device__ __forceinline__ uint64_t fingerprint_mix(uint64_t x) {
    x^=x>>30; x*=0xbf58476d1ce4e5b9ULL;
    x^=x>>27; x*=0x94d049bb133111ebULL;
    return x^(x>>31);
}
__device__ __forceinline__ void mini_fingerprint(const uint8_t* key,uint64_t* low,uint64_t* middle,uint64_t* high) {
    uint64_t a=0xcbf29ce484222325ULL, b=0x84222325cbf29ce4ULL, c=0x9e3779b97f4a7c15ULL;
    #pragma unroll
    for(int i=0;i<33;i++) {
        a=(a^(uint64_t)key[i])*0x100000001b3ULL;
        b=(b^((uint64_t)key[i]+(uint64_t)i*0x9e3779b97f4a7c15ULL))*0x100000001b3ULL;
        c=(c^((uint64_t)key[i]+(uint64_t)i*0xd6e8feb86659fd93ULL))*0x100000001b3ULL;
    }
    *low=fingerprint_mix(a);
    *middle=fingerprint_mix(b^a);
    *high=fingerprint_mix(c^b);
}
__device__ __forceinline__ bool fingerprint_equal(const uint8_t* stored,uint64_t low,uint64_t middle,uint64_t high,uint32_t fingerprint_bytes) {
    uint64_t words[3]={low,middle,high};
    const uint8_t* expected=(const uint8_t*)words;
    for(uint32_t i=0;i<fingerprint_bytes;i++) if(stored[i]!=expected[i]) return false;
    return true;
}
__device__ bool table_lookup(const uint8_t* fingerprints,const uint64_t* indices,uint64_t capacity,uint32_t fingerprint_bytes,const uint64_t* bloom,uint64_t bloom_words,uint32_t bloom_enabled,const uint64_t* rejected_indices,uint64_t rejected_count,bool reject_first_point,const uint8_t* key,uint64_t* mini_index) {
    if(bloom_enabled) {
        uint64_t h1=0,h2=0;
        #pragma unroll
        for(int i=0;i<8;i++) { h1|=(uint64_t)key[i+1]<<(8*i); h2|=(uint64_t)key[i+9]<<(8*i); }
        h2|=1ULL;
        uint64_t mixed=((h1<<23)|(h1>>41))^((h2>>11)|(h2<<53));
        uint64_t word=bloom[h1&(bloom_words-1)];
        #pragma unroll
        for(int probe=0;probe<6;probe++) {
            uint32_t bit=(uint32_t)((mixed+(uint64_t)probe*h2)&63ULL);
            if((word&(1ULL<<bit))==0) return false;
        }
    }
    uint64_t fingerprint_low,fingerprint_middle,fingerprint_high;
    mini_fingerprint(key,&fingerprint_low,&fingerprint_middle,&fingerprint_high);
    uint64_t slot=__umul64hi((unsigned long long)fingerprint_low,(unsigned long long)capacity);
    for(uint64_t probe=0;probe<capacity;probe++){
        const uint8_t* entry=fingerprints+slot*fingerprint_bytes;
        uint64_t encoded_index=indices[slot];
        if(encoded_index==0) return false;
        if(encoded_index!=0xffffffffffffffffULL) {
            if(fingerprint_equal(entry,fingerprint_low,fingerprint_middle,fingerprint_high,fingerprint_bytes)){
                uint64_t index=encoded_index-1;
                bool rejected=false;
                if(reject_first_point) for(uint64_t i=0;i<rejected_count;i++) rejected|=rejected_indices[i]==index;
                if(!rejected) { *mini_index=index; return true; }
            }
        }
        if(++slot==capacity) slot=0;
    }
    return false;
}

__device__ void insert_generated_mini(uint8_t* fingerprints_table,uint64_t* indices_table,uint64_t capacity,uint64_t* bloom,uint64_t bloom_word_count,uint32_t fingerprint_bytes,const uint8_t* key,uint64_t index) {
    uint64_t fingerprint_low,fingerprint_middle,fingerprint_high;
    mini_fingerprint(key,&fingerprint_low,&fingerprint_middle,&fingerprint_high);
    uint64_t h1=0,h2=0;
    #pragma unroll
    for(int i=0;i<8;i++) { h1|=(uint64_t)key[i+1]<<(8*i); h2|=(uint64_t)key[i+9]<<(8*i); }
    h2|=1ULL;
    uint64_t mixed=((h1<<23)|(h1>>41))^((h2>>11)|(h2<<53));
    uint64_t bloom_block=h1&(bloom_word_count-1);
    #pragma unroll
    for(int probe=0;probe<6;probe++) {
        uint32_t bit=(uint32_t)((mixed+(uint64_t)probe*h2)&63ULL);
        atomicOr((unsigned long long*)&bloom[bloom_block],1ULL<<bit);
    }
    uint64_t slot=__umul64hi((unsigned long long)fingerprint_low,(unsigned long long)capacity);
    for(uint64_t probe=0;probe<capacity;probe++) {
        uint8_t* destination=fingerprints_table+slot*fingerprint_bytes;
        unsigned long long* index_word=(unsigned long long*)&indices_table[slot];
        unsigned long long observed=atomicCAS(index_word,0ULL,0xffffffffffffffffULL);
        if(observed==0ULL) {
            uint64_t fingerprints[3]={fingerprint_low,fingerprint_middle,fingerprint_high};
            const uint8_t* fingerprint_bytes_source=(const uint8_t*)fingerprints;
            for(uint32_t byte=0;byte<fingerprint_bytes;byte++) destination[byte]=fingerprint_bytes_source[byte];
            __threadfence();
            atomicExch(index_word,(unsigned long long)(index+1ULL));
            return;
        }
        if(observed==0xffffffffffffffffULL) {
            if(++slot==capacity) slot=0;
            continue;
        }
        if(fingerprint_equal(destination,fingerprint_low,fingerprint_middle,fingerprint_high,fingerprint_bytes) && observed==index+1ULL) return;
        if(++slot==capacity) slot=0;
    }
}

extern "C" __global__ void insert_minis_kernel(uint8_t* fingerprints_table,uint64_t* indices_table,uint64_t capacity,uint64_t* bloom,uint64_t bloom_word_count,uint32_t fingerprint_bytes,const uint8_t* delta_xy,uint32_t chunk_size,uint64_t mini_count,const uint8_t* seeds,uint64_t seed_count) {
    uint64_t tid=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(tid>=seed_count) return;
    const uint8_t* seed=seeds+tid*72;
    uint64_t start_index=0;
    #pragma unroll
    for(int i=0;i<8;i++) start_index|=(uint64_t)seed[64+i]<<(8*i);
    if(start_index>=mini_count) return;
    uint64_t remaining=mini_count-start_index;
    uint32_t local_count=(uint32_t)min((uint64_t)chunk_size,remaining);
    Point point=Point{load_be256(seed),load_be256(seed+32),one_u256(),false};
    Point delta=Point{load_be256(delta_xy),load_be256(delta_xy+32),one_u256(),false};
    for(uint32_t i=0;i<local_count;i++) {
        uint8_t compressed[33];
        serialize_affine(point.x,point.y,compressed);
        insert_generated_mini(fingerprints_table,indices_table,capacity,bloom,bloom_word_count,fingerprint_bytes,compressed,start_index+i);
        if(i+1<local_count) point=affine_add_points(point,delta);
    }
}
}

extern "C" __global__ void secp256k1_search_kernel(
    uint8_t* worker_points_xy,
    const uint8_t* step_xy,
    uint32_t* worker_adjustments,
    const uint32_t* step_scalar,
    uint64_t count,
    uint64_t worker_count,
    uint64_t chunk_size,
    const uint8_t* mini_fingerprints,
    const uint64_t* mini_indices,
    uint64_t mini_capacity,
    uint32_t fingerprint_bytes,
    const uint64_t* bloom_filter,
    uint64_t bloom_word_count,
    uint32_t bloom_enabled,
    const uint32_t* mini_start_limbs,
    const uint32_t* mini_delta_limbs,
    int32_t direction,
    uint32_t first_batch,
    const uint8_t* batch_gap_xy,
    const uint32_t* batch_gap_scalar,
    const uint64_t* rejected_mini_indices,
    uint64_t rejected_count,
    uint32_t* any_candidate,
    uint32_t* worker_results) {
    uint64_t tid=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    bool valid_worker=tid<worker_count;
    uint64_t begin=tid*chunk_size;
    bool active=valid_worker && begin<count;
    uint64_t local_count=active?min(chunk_size,count-begin):0;
    Point point=infinity_point();
    U256 adjustment=zero_u256(), scalar_step;
    if(valid_worker) {
        uint8_t* xy=worker_points_xy+tid*64;
        point=Point{load_be256(xy),load_be256(xy+32),one_u256(),false};
        #pragma unroll
        for(int i=0;i<8;i++) adjustment.v[i]=worker_adjustments[tid*8+i];
    }
    #pragma unroll
    for(int i=0;i<8;i++) scalar_step.v[i]=step_scalar[i];
    U256 dx=load_be256(step_xy), dy=load_be256(step_xy+32);
    if(active && !first_batch) {
        U256 gap_x=load_be256(batch_gap_xy), gap_y=load_be256(batch_gap_xy+32);
        point=affine_add_points(point,Point{gap_x,gap_y,one_u256(),false});
        U256 gap;
        #pragma unroll
        for(int i=0;i<8;i++) gap.v[i]=batch_gap_scalar[i];
        adjustment=scalar_add(adjustment,gap);
    }

    for(uint64_t i=0;i<chunk_size;i++) {
        active=active && i<local_count;
        if(active && !point.infinity) {
            uint8_t compressed[33]; serialize_affine(point.x,point.y,compressed);
            uint64_t mini_index;
            uint64_t global_index=begin+i;
            if(table_lookup(mini_fingerprints,mini_indices,mini_capacity,fingerprint_bytes,bloom_filter,bloom_word_count,bloom_enabled,rejected_mini_indices,rejected_count,global_index==0,compressed,&mini_index)) {
                U256 mini_start,mini_delta;
                #pragma unroll
                for(int j=0;j<8;j++) { mini_start.v[j]=mini_start_limbs[j]; mini_delta.v[j]=mini_delta_limbs[j]; }
                U256 mini_private=scalar_add(mini_start,scalar_mul_u64(mini_delta,mini_index));
                U256 private_key=direction>0?scalar_sub(mini_private,adjustment):scalar_add(mini_private,adjustment);
                uint32_t* result=worker_results+tid*12;
                result[0]=1;
                atomicExch(any_candidate,1U);
                #pragma unroll
                for(int j=0;j<8;j++) result[1+j]=private_key.v[j];
                result[9]=(uint32_t)i;
                result[10]=(uint32_t)mini_index;
                result[11]=(uint32_t)(mini_index>>32);
                active=false;
            }
        }
        bool advance=active && i+1<local_count && !point.infinity;
#ifdef WARP_BATCH_INVERSION
        U256 denominator=one_u256(), numerator=zero_u256();
        if(advance) {
            if(equal_u256(point.x,dx)) {
                if(equal_u256(point.y,dy) && !is_zero(point.y)) {
                    U256 x2=fmul(point.x,point.x);
                    numerator=fadd(x2,fadd(x2,x2));
                    denominator=fadd(point.y,point.y);
                } else {
                    point=infinity_point();
                    advance=false;
                }
            } else {
                numerator=fsub(dy,point.y);
                denominator=fsub(dx,point.x);
            }
        }
        U256 denominator_inverse=warp_batch_inverse(denominator);
        if(advance) {
            U256 slope=fmul(numerator,denominator_inverse);
            U256 next_x=fsub(fsub(fmul(slope,slope),point.x),dx);
            U256 next_y=fsub(fmul(slope,fsub(point.x,next_x)),point.y);
            point=Point{next_x,next_y,one_u256(),false};
            adjustment=scalar_add(adjustment,scalar_step);
        }
#else
        if(advance) {
            point=affine_add_points(point,Point{dx,dy,one_u256(),false});
            adjustment=scalar_add(adjustment,scalar_step);
        }
#endif
    }
    if(valid_worker && active) {
        uint8_t* xy=worker_points_xy+tid*64;
        store_be256(point.x,xy);
        store_be256(point.y,xy+32);
        #pragma unroll
        for(int i=0;i<8;i++) worker_adjustments[tid*8+i]=adjustment.v[i];
    }
}
