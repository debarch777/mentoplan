import * as THREE from 'three'

// 3D simplex noise (Stefan Gustavson / Ashima Arts, MIT license)
const NOISE = /* glsl */ `
vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+10.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0); const vec4 D=vec4(0.0,0.5,1.0,2.0);
  vec3 i=floor(v+dot(v,C.yyy)); vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz); vec3 l=1.0-g; vec3 i1=min(g.xyz,l.zxy); vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx; vec3 x2=x0-i2+C.yyy; vec3 x3=x0-D.yyy;
  i=mod289(i);
  vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
  float n_=0.142857142857; vec3 ns=n_*D.wyz-D.xzx;
  vec4 j=p-49.0*floor(p*ns.z*ns.z); vec4 x_=floor(j*ns.z); vec4 y_=floor(j-7.0*x_);
  vec4 x=x_*ns.x+ns.yyyy; vec4 y=y_*ns.x+ns.yyyy; vec4 h=1.0-abs(x)-abs(y);
  vec4 b0=vec4(x.xy,y.xy); vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.0+1.0; vec4 s1=floor(b1)*2.0+1.0; vec4 sh=-step(h,vec4(0.0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy; vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x); vec3 p1=vec3(a0.zw,h.y); vec3 p2=vec3(a1.xy,h.z); vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
  vec4 m=max(0.5-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0); m=m*m;
  return 105.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}
`

export interface SkinParams {
  tone: THREE.Color
  lip: THREE.Color
  beard: number
  sss: number
  detail: number
  oil: number
  overlay: 0 | 1 | 2 // 0 none, 1 uncertainty band width, 2 change (normal displacement)
  overlayScale: number
  hasTexture: boolean
}

export const SKIN_TONES: Record<string, string> = {
  'I – very fair': '#f1d3c2',
  'II – fair': '#e6bea3',
  'III – medium': '#d29f80',
  'IV – olive': '#b47c5a',
  'V – brown': '#8a573a',
  'VI – dark brown': '#5a3726',
}

export function makeSkinMaterial(params: SkinParams, texture?: THREE.Texture | null) {
  const mat = new THREE.MeshPhysicalMaterial({
    color: 0xffffff,
    roughness: 0.52,
    metalness: 0,
    ior: 1.4,
    specularIntensity: 0.55,
    clearcoat: 0.18,
    clearcoatRoughness: 0.32,
    sheen: 0.35,
    sheenRoughness: 0.55,
    sheenColor: new THREE.Color('#f3d9cf'),
    map: texture ?? null,
    side: THREE.DoubleSide,
  })
  const uniforms = {
    uTone: { value: params.tone.clone() },
    uLip: { value: params.lip.clone() },
    uBeard: { value: params.beard },
    uSSS: { value: params.sss },
    uDetail: { value: params.detail },
    uOverlay: { value: params.overlay },
    uOverlayScale: { value: params.overlayScale },
    uHasTex: { value: params.hasTexture ? 1 : 0 },
  }
  mat.userData.uniforms = uniforms
  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms)
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', `#include <common>
        attribute float aUncert; attribute float aChange; attribute float aLip; attribute float aBeard; attribute float aCavity; attribute vec3 aAlbedo; attribute float aTexMask; attribute float aMouth; varying float vMouth;
        varying vec3 vAlbedo; varying float vTexMask;
        varying vec3 vObj; varying float vUncert; varying float vChange; varying float vLip; varying float vBeard; varying float vCavity;`)
      .replace('#include <begin_vertex>', `#include <begin_vertex>
        vObj = position; vUncert = aUncert; vChange = aChange; vLip = aLip; vBeard = aBeard; vCavity = aCavity; vAlbedo = aAlbedo; vTexMask = aTexMask; vMouth = aMouth;`)
    shader.fragmentShader = shader.fragmentShader
      .replace('#include <common>', `#include <common>
        uniform vec3 uTone; uniform vec3 uLip; uniform float uBeard; uniform float uSSS; uniform float uDetail;
        uniform int uOverlay; uniform float uOverlayScale; uniform int uHasTex;
        varying vec3 vObj; varying float vUncert; varying float vChange; varying float vLip; varying float vBeard; varying float vCavity; varying vec3 vAlbedo; varying float vTexMask; varying float vMouth;
        ${NOISE}
        vec3 heat(float t){ t=clamp(t,0.0,1.0); return mix(mix(vec3(0.05,0.2,0.6),vec3(0.1,0.75,0.55),smoothstep(0.0,0.4,t)),
                                     mix(vec3(0.98,0.85,0.2),vec3(0.85,0.12,0.1),smoothstep(0.6,1.0,t)),smoothstep(0.35,0.7,t)); }
        vec3 diverge(float t){ t=clamp(t,-1.0,1.0); return t<0.0? mix(vec3(0.92),vec3(0.1,0.35,0.85),-t) : mix(vec3(0.92),vec3(0.85,0.15,0.1),t); }
        float skinHeight(vec3 p){
          // pores (~0.35 mm) + fine relief (~1 mm) + broad undulation; units are millimetres
          float pores = 1.0 - smoothstep(0.15, 0.75, abs(snoise(p*2.9)));
          float fine = snoise(p*1.1) * 0.35 + snoise(p*5.5) * 0.15;
          return (pores * 0.6 + fine) * 0.035;
        }
        vec3 perturbSkin(vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection) {
          vec3 vSigmaX = normalize(dFdx(surf_pos)); vec3 vSigmaY = normalize(dFdy(surf_pos));
          vec3 vN = surf_norm; vec3 R1 = cross(vSigmaY, vN); vec3 R2 = cross(vN, vSigmaX);
          float fDet = dot(vSigmaX, R1) * faceDirection;
          vec3 vGrad = sign(fDet) * (dHdxy.x * R1 + dHdxy.y * R2);
          return normalize(abs(fDet) * surf_norm - vGrad);
        }`)
      .replace('#include <clipping_planes_fragment>', `#include <clipping_planes_fragment>
        if (vCavity > 0.5) discard;   // oral fissure opened by an expression
        `)
      .replace('#include <map_fragment>', `#include <map_fragment>
        {
          vec3 p = vObj;
          vec3 base = uHasTex == 1 ? diffuseColor.rgb : uTone;
          float n1 = snoise(p*0.05); float n2 = snoise(p*0.32); float n3 = snoise(p*1.7);
          // melanin / haemoglobin mottling
          vec3 alb = base * (1.0 + 0.05*n1 + 0.035*n2 + 0.02*n3);
          alb = mix(alb, alb*vec3(1.06,0.93,0.92), 0.25*smoothstep(0.2,0.9,n1));
          // photographic albedo from a registered face scan (where available)
          alb = mix(alb, vAlbedo * (1.0 + 0.015*n3), vTexMask);
          if (uHasTex == 0) {
            alb = mix(alb, uLip * (0.92 + 0.08*n3), vLip * (1.0 - vTexMask));
            float stubble = smoothstep(0.1, 0.9, snoise(p*3.6)*0.5+0.5);
            alb *= 1.0 - uBeard * vBeard * (1.0 - vTexMask) * (0.18 + 0.22*stubble);
          }
          if (uOverlay == 1) alb = mix(alb, heat(vUncert / uOverlayScale), 0.78);
          if (uOverlay == 2) alb = mix(alb, diverge(vChange / uOverlayScale), smoothstep(0.02, 0.4, abs(vChange)/uOverlayScale) * 0.85);
          // inside of the head seen through the opened mouth: dark oral mucosa
          if (!gl_FrontFacing) alb = mix(vec3(0.085, 0.075, 0.07), vec3(0.16, 0.035, 0.035), vMouth);
          diffuseColor.rgb = alb;
        }`)
      .replace('#include <roughnessmap_fragment>', `#include <roughnessmap_fragment>
        roughnessFactor = clamp(roughnessFactor * (1.0 + 0.12*snoise(vObj*0.9)) * mix(1.0, 0.72, vLip), 0.18, 0.95);`)
      .replace('#include <normal_fragment_maps>', `#include <normal_fragment_maps>
        {
          float h = skinHeight(vObj) * uDetail;
          vec2 dHdxy = vec2(dFdx(h), dFdy(h));
          normal = perturbSkin(-vViewPosition, normal, dHdxy, faceDirection);
        }`)
      .replace('#include <lights_physical_pars_fragment>', (THREE.ShaderChunk.lights_physical_pars_fragment as string)
        .replace('reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );', `
          {
            // subsurface approximation: per-channel wrapped diffuse (red scatters furthest)
            float ndl = dot( geometryNormal, directLight.direction );
            vec3 w = vec3(0.55, 0.26, 0.17) * uSSS;
            vec3 wrapped = clamp( (vec3(ndl) + w) / ( (1.0 + w) * (1.0 + w) ), 0.0, 1.0 );
            reflectedLight.directDiffuse += wrapped * directLight.color * BRDF_Lambert( material.diffuseColor );
          }`))
  }
  mat.customProgramCacheKey = () => 'mentoplan-skin-v4'
  return mat
}

export function updateSkinParams(mat: THREE.MeshPhysicalMaterial, p: Partial<SkinParams>) {
  const u = mat.userData.uniforms
  if (!u) return
  if (p.tone) u.uTone.value.copy(p.tone)
  if (p.lip) u.uLip.value.copy(p.lip)
  if (p.beard !== undefined) u.uBeard.value = p.beard
  if (p.sss !== undefined) u.uSSS.value = p.sss
  if (p.detail !== undefined) u.uDetail.value = p.detail
  if (p.overlay !== undefined) u.uOverlay.value = p.overlay
  if (p.overlayScale !== undefined) u.uOverlayScale.value = p.overlayScale
  if (p.oil !== undefined) {
    mat.clearcoat = 0.05 + 0.35 * p.oil
    mat.roughness = 0.6 - 0.15 * p.oil
  }
}
