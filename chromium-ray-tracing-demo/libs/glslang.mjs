
var Module = (function() {
  var _scriptDir = typeof document !== 'undefined' && document.currentScript ? document.currentScript.src : undefined;
  
  return (
function(Module) {
  Module = Module || {};


var c;c||(c=typeof Module !== 'undefined' ? Module : {});
c.compileGLSLZeroCopy=function(a,d,e,f){e=!!e;switch(d){case "vertex":var g=0;break;case "fragment":g=4;break;case "compute":g=5;break;case "raygen":g=6;break;case "intersect":g=7;break;case "anyhit":g=8;break;case "closesthit":g=9;break;case "miss":g=10;break;case "callable":g=11;break;default:throw Error("shader_stage must be 'vertex', 'fragment', or 'compute'.");}switch(f||"1.0"){case "1.0":var h=65536;break;case "1.1":h=65792;break;case "1.2":h=66048;break;case "1.3":h=66304;break;case "1.4":h=
66560;break;case "1.5":h=66816;break;default:throw Error("spirv_version must be '1.0' ~ '1.5'.");}f=c._malloc(4);d=c._malloc(4);var l=aa([a,g,e,h,f,d]);e=m(f);a=m(d);c._free(f);c._free(d);if(0===l)throw Error("GLSL compilation failed");f={};e/=4;f.data=c.HEAPU32.subarray(e,e+a);f.free=function(){c._destroy_output_buffer(l)};return f};c.compileGLSL=function(a,d,e,f){a=c.compileGLSLZeroCopy(a,d,e,f);d=a.data.slice();a.free();return d};var u={},v;for(v in c)c.hasOwnProperty(v)&&(u[v]=c[v]);
var w="./this.program",x=!1,y=!1;x="object"===typeof window;y="function"===typeof importScripts;var z="",A;if(x||y)y?z=self.location.href:document.currentScript&&(z=document.currentScript.src),_scriptDir&&(z=_scriptDir),0!==z.indexOf("blob:")?z=z.substr(0,z.lastIndexOf("/")+1):z="",y&&(A=function(a){try{var d=new XMLHttpRequest;d.open("GET",a,!1);d.responseType="arraybuffer";d.send(null);return new Uint8Array(d.response)}catch(e){if(a=ba(a))return a;throw e;}});
var ca=c.print||console.log.bind(console),B=c.printErr||console.warn.bind(console);for(v in u)u.hasOwnProperty(v)&&(c[v]=u[v]);u=null;c.thisProgram&&(w=c.thisProgram);var D;c.wasmBinary&&(D=c.wasmBinary);var noExitRuntime;c.noExitRuntime&&(noExitRuntime=c.noExitRuntime);"object"!==typeof WebAssembly&&B("no native wasm support detected");
function m(a){var d="i32";"*"===d.charAt(d.length-1)&&(d="i32");switch(d){case "i1":return E[a>>0];case "i8":return E[a>>0];case "i16":return da[a>>1];case "i32":return F[a>>2];case "i64":return F[a>>2];case "float":return ea[a>>2];case "double":return fa[a>>3];default:G("invalid type for getValue: "+d)}return null}var H,ha=new WebAssembly.Table({initial:859,maximum:859,element:"anyfunc"}),ia=!1;
function ja(){var a=c._convert_glsl_to_spirv;a||G("Assertion failed: Cannot call unknown function convert_glsl_to_spirv, make sure it is exported");return a}
function aa(a){var d="string number boolean number number number".split(" "),e={string:function(r){var n=0;if(null!==r&&void 0!==r&&0!==r){var q=(r.length<<2)+1;n=I(q);ka(r,J,n,q)}return n},array:function(r){var n=I(r.length);E.set(r,n);return n}},f=ja(),g=[],h=0;if(a)for(var l=0;l<a.length;l++){var t=e[d[l]];t?(0===h&&(h=la()),g[l]=t(a[l])):g[l]=a[l]}a=f.apply(null,g);0!==h&&ma(h);return a}var na="undefined"!==typeof TextDecoder?new TextDecoder("utf8"):void 0;
function K(a,d,e){var f=d+e;for(e=d;a[e]&&!(e>=f);)++e;if(16<e-d&&a.subarray&&na)return na.decode(a.subarray(d,e));for(f="";d<e;){var g=a[d++];if(g&128){var h=a[d++]&63;if(192==(g&224))f+=String.fromCharCode((g&31)<<6|h);else{var l=a[d++]&63;g=224==(g&240)?(g&15)<<12|h<<6|l:(g&7)<<18|h<<12|l<<6|a[d++]&63;65536>g?f+=String.fromCharCode(g):(g-=65536,f+=String.fromCharCode(55296|g>>10,56320|g&1023))}}else f+=String.fromCharCode(g)}return f}
function ka(a,d,e,f){if(0<f){f=e+f-1;for(var g=0;g<a.length;++g){var h=a.charCodeAt(g);if(55296<=h&&57343>=h){var l=a.charCodeAt(++g);h=65536+((h&1023)<<10)|l&1023}if(127>=h){if(e>=f)break;d[e++]=h}else{if(2047>=h){if(e+1>=f)break;d[e++]=192|h>>6}else{if(65535>=h){if(e+2>=f)break;d[e++]=224|h>>12}else{if(e+3>=f)break;d[e++]=240|h>>18;d[e++]=128|h>>12&63}d[e++]=128|h>>6&63}d[e++]=128|h&63}}d[e]=0}}"undefined"!==typeof TextDecoder&&new TextDecoder("utf-16le");var L,E,J,da,F,ea,fa;
function oa(a){L=a;c.HEAP8=E=new Int8Array(a);c.HEAP16=da=new Int16Array(a);c.HEAP32=F=new Int32Array(a);c.HEAPU8=J=new Uint8Array(a);c.HEAPU16=new Uint16Array(a);c.HEAPU32=new Uint32Array(a);c.HEAPF32=ea=new Float32Array(a);c.HEAPF64=fa=new Float64Array(a)}var pa=c.INITIAL_MEMORY||16777216;c.wasmMemory?H=c.wasmMemory:H=new WebAssembly.Memory({initial:pa/65536,maximum:32768});H&&(L=H.buffer);pa=L.byteLength;oa(L);F[87572]=5593328;
function M(a){for(;0<a.length;){var d=a.shift();if("function"==typeof d)d(c);else{var e=d.I;"number"===typeof e?void 0===d.H?c.dynCall_v(e):c.dynCall_vi(e,d.H):e(void 0===d.H?null:d.H)}}}var qa=[],ra=[],sa=[],ta=[];function ua(){var a=c.preRun.shift();qa.unshift(a)}var N=0,O=null,P=null;c.preloadedImages={};c.preloadedAudios={};function G(a){if(c.onAbort)c.onAbort(a);ca(a);B(a);ia=!0;throw new WebAssembly.RuntimeError("abort("+a+"). Build with -s ASSERTIONS=1 for more info.");}
function Ba(){return D||!x&&!y||"function"!==typeof fetch?new Promise(function(a){a(Aa())}):fetch(R,{credentials:"same-origin"}).then(function(a){if(!a.ok)throw"failed to load wasm binary file at '"+R+"'";return a.arrayBuffer()}).catch(function(){return Aa()})}ra.push({I:function(){Ca()}});var Da={},Ea=[null,[],[]],Fa={};
function Ga(){if(!S){var a={USER:"web_user",LOGNAME:"web_user",PATH:"/",PWD:"/",HOME:"/home/web_user",LANG:("object"===typeof navigator&&navigator.languages&&navigator.languages[0]||"C").replace("-","_")+".UTF-8",_:w||"./this.program"},d;for(d in Fa)a[d]=Fa[d];var e=[];for(d in a)e.push(d+"="+a[d]);S=e}return S}var S;function T(a){return 0===a%4&&(0!==a%100||0===a%400)}function U(a,d){for(var e=0,f=0;f<=d;e+=a[f++]);return e}
var V=[31,29,31,30,31,30,31,31,30,31,30,31],W=[31,28,31,30,31,30,31,31,30,31,30,31];function X(a,d){for(a=new Date(a.getTime());0<d;){var e=a.getMonth(),f=(T(a.getFullYear())?V:W)[e];if(d>f-a.getDate())d-=f-a.getDate()+1,a.setDate(1),11>e?a.setMonth(e+1):(a.setMonth(0),a.setFullYear(a.getFullYear()+1));else{a.setDate(a.getDate()+d);break}}return a}
function Ha(a,d,e,f){function g(b,k,p){for(b="number"===typeof b?b.toString():b||"";b.length<k;)b=p[0]+b;return b}function h(b,k){return g(b,k,"0")}function l(b,k){function p(xa){return 0>xa?-1:0<xa?1:0}var C;0===(C=p(b.getFullYear()-k.getFullYear()))&&0===(C=p(b.getMonth()-k.getMonth()))&&(C=p(b.getDate()-k.getDate()));return C}function t(b){switch(b.getDay()){case 0:return new Date(b.getFullYear()-1,11,29);case 1:return b;case 2:return new Date(b.getFullYear(),0,3);case 3:return new Date(b.getFullYear(),
0,2);case 4:return new Date(b.getFullYear(),0,1);case 5:return new Date(b.getFullYear()-1,11,31);case 6:return new Date(b.getFullYear()-1,11,30)}}function r(b){b=X(new Date(b.A+1900,0,1),b.G);var k=new Date(b.getFullYear()+1,0,4),p=t(new Date(b.getFullYear(),0,4));k=t(k);return 0>=l(p,b)?0>=l(k,b)?b.getFullYear()+1:b.getFullYear():b.getFullYear()-1}var n=F[f+40>>2];f={L:F[f>>2],K:F[f+4>>2],D:F[f+8>>2],C:F[f+12>>2],B:F[f+16>>2],A:F[f+20>>2],F:F[f+24>>2],G:F[f+28>>2],R:F[f+32>>2],J:F[f+36>>2],M:n?n?
K(J,n,void 0):"":""};e=e?K(J,e,void 0):"";n={"%c":"%a %b %d %H:%M:%S %Y","%D":"%m/%d/%y","%F":"%Y-%m-%d","%h":"%b","%r":"%I:%M:%S %p","%R":"%H:%M","%T":"%H:%M:%S","%x":"%m/%d/%y","%X":"%H:%M:%S","%Ec":"%c","%EC":"%C","%Ex":"%m/%d/%y","%EX":"%H:%M:%S","%Ey":"%y","%EY":"%Y","%Od":"%d","%Oe":"%e","%OH":"%H","%OI":"%I","%Om":"%m","%OM":"%M","%OS":"%S","%Ou":"%u","%OU":"%U","%OV":"%V","%Ow":"%w","%OW":"%W","%Oy":"%y"};for(var q in n)e=e.replace(new RegExp(q,"g"),n[q]);var ya="Sunday Monday Tuesday Wednesday Thursday Friday Saturday".split(" "),
za="January February March April May June July August September October November December".split(" ");n={"%a":function(b){return ya[b.F].substring(0,3)},"%A":function(b){return ya[b.F]},"%b":function(b){return za[b.B].substring(0,3)},"%B":function(b){return za[b.B]},"%C":function(b){return h((b.A+1900)/100|0,2)},"%d":function(b){return h(b.C,2)},"%e":function(b){return g(b.C,2," ")},"%g":function(b){return r(b).toString().substring(2)},"%G":function(b){return r(b)},"%H":function(b){return h(b.D,2)},
"%I":function(b){b=b.D;0==b?b=12:12<b&&(b-=12);return h(b,2)},"%j":function(b){return h(b.C+U(T(b.A+1900)?V:W,b.B-1),3)},"%m":function(b){return h(b.B+1,2)},"%M":function(b){return h(b.K,2)},"%n":function(){return"\n"},"%p":function(b){return 0<=b.D&&12>b.D?"AM":"PM"},"%S":function(b){return h(b.L,2)},"%t":function(){return"\t"},"%u":function(b){return b.F||7},"%U":function(b){var k=new Date(b.A+1900,0,1),p=0===k.getDay()?k:X(k,7-k.getDay());b=new Date(b.A+1900,b.B,b.C);return 0>l(p,b)?h(Math.ceil((31-
p.getDate()+(U(T(b.getFullYear())?V:W,b.getMonth()-1)-31)+b.getDate())/7),2):0===l(p,k)?"01":"00"},"%V":function(b){var k=new Date(b.A+1901,0,4),p=t(new Date(b.A+1900,0,4));k=t(k);var C=X(new Date(b.A+1900,0,1),b.G);return 0>l(C,p)?"53":0>=l(k,C)?"01":h(Math.ceil((p.getFullYear()<b.A+1900?b.G+32-p.getDate():b.G+1-p.getDate())/7),2)},"%w":function(b){return b.F},"%W":function(b){var k=new Date(b.A,0,1),p=1===k.getDay()?k:X(k,0===k.getDay()?1:7-k.getDay()+1);b=new Date(b.A+1900,b.B,b.C);return 0>l(p,
b)?h(Math.ceil((31-p.getDate()+(U(T(b.getFullYear())?V:W,b.getMonth()-1)-31)+b.getDate())/7),2):0===l(p,k)?"01":"00"},"%y":function(b){return(b.A+1900).toString().substring(2)},"%Y":function(b){return b.A+1900},"%z":function(b){b=b.J;var k=0<=b;b=Math.abs(b)/60;return(k?"+":"-")+String("0000"+(b/60*100+b%60)).slice(-4)},"%Z":function(b){return b.M},"%%":function(){return"%"}};for(q in n)0<=e.indexOf(q)&&(e=e.replace(new RegExp(q,"g"),n[q](f)));q=Ia(e);if(q.length>d)return 0;E.set(q,a);return q.length-
1}function Ia(a){for(var d=0,e=0;e<a.length;++e){var f=a.charCodeAt(e);55296<=f&&57343>=f&&(f=65536+((f&1023)<<10)|a.charCodeAt(++e)&1023);127>=f?++d:d=2047>=f?d+2:65535>=f?d+3:d+4}d=Array(d+1);ka(a,d,0,d.length);return d}
var Ja="function"===typeof atob?atob:function(a){var d="",e=0;a=a.replace(/[^A-Za-z0-9\+\/=]/g,"");do{var f="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".indexOf(a.charAt(e++));var g="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".indexOf(a.charAt(e++));var h="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".indexOf(a.charAt(e++));var l="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=".indexOf(a.charAt(e++));f=f<<2|g>>4;
g=(g&15)<<4|h>>2;var t=(h&3)<<6|l;d+=String.fromCharCode(f);64!==h&&(d+=String.fromCharCode(g));64!==l&&(d+=String.fromCharCode(t))}while(e<a.length);return d};function ba(a){if(Q(a)){try{var d=Ja(a.slice(va.length)),e=new Uint8Array(d.length);for(a=0;a<d.length;++a)e[a]=d.charCodeAt(a)}catch(f){throw Error("Converting base64 string to bytes failed.");}return e}}
var Ma={e:function(){F[Ka()>>2]=63;return-1},d:function(a,d){if(-1===(a|0)||0===d)a=-28;else{var e=Da[a];e&&d===e.O&&(Da[a]=null,e.N&&La(e.P));a=0}return a},b:function(){G()},k:function(a,d,e){J.copyWithin(a,d,d+e)},l:function(a){var d=J.length;if(2147483648<a)return!1;for(var e=1;4>=e;e*=2){var f=d*(1+.2/e);f=Math.min(f,a+100663296);f=Math.max(16777216,a,f);0<f%65536&&(f+=65536-f%65536);a:{try{H.grow(Math.min(2147483648,f)-L.byteLength+65535>>>16);oa(H.buffer);var g=1;break a}catch(h){}g=void 0}if(g)return!0}return!1},
f:function(a,d){var e=0;Ga().forEach(function(f,g){var h=d+e;g=F[a+4*g>>2]=h;for(h=0;h<f.length;++h)E[g++>>0]=f.charCodeAt(h);E[g>>0]=0;e+=f.length+1});return 0},g:function(a,d){var e=Ga();F[a>>2]=e.length;var f=0;e.forEach(function(g){f+=g.length+1});F[d>>2]=f;return 0},h:function(){return 0},j:function(){},a:function(a,d,e,f){for(var g=0,h=0;h<e;h++){for(var l=F[d+8*h>>2],t=F[d+(8*h+4)>>2],r=0;r<t;r++){var n=J[l+r],q=Ea[a];0===n||10===n?((1===a?ca:B)(K(q,0)),q.length=0):q.push(n)}g+=t}F[f>>2]=g;
return 0},memory:H,m:function(){},i:function(){},c:function(a,d,e,f){return Ha(a,d,e,f)},table:ha},Na=function(){function a(g){c.asm=g.exports;N--;c.monitorRunDependencies&&c.monitorRunDependencies(N);0==N&&(null!==O&&(clearInterval(O),O=null),P&&(g=P,P=null,g()))}function d(g){a(g.instance)}function e(g){return Ba().then(function(h){return WebAssembly.instantiate(h,f)}).then(g,function(h){B("failed to asynchronously prepare wasm: "+h);G(h)})}var f={a:Ma};N++;c.monitorRunDependencies&&c.monitorRunDependencies(N);
if(c.instantiateWasm)try{return c.instantiateWasm(f,a)}catch(g){return B("Module.instantiateWasm callback failed with error: "+g),!1}(function(){if(D||"function"!==typeof WebAssembly.instantiateStreaming||Q(R)||"function"!==typeof fetch)return e(d);fetch(R,{credentials:"same-origin"}).then(function(g){return WebAssembly.instantiateStreaming(g,f).then(d,function(h){B("wasm streaming compile failed: "+h);B("falling back to ArrayBuffer instantiation");e(d)})})})();return{}}();c.asm=Na;
var Ca=c.___wasm_call_ctors=function(){return(Ca=c.___wasm_call_ctors=c.asm.n).apply(null,arguments)};c._convert_glsl_to_spirv=function(){return(c._convert_glsl_to_spirv=c.asm.o).apply(null,arguments)};c._destroy_output_buffer=function(){return(c._destroy_output_buffer=c.asm.p).apply(null,arguments)};c._malloc=function(){return(c._malloc=c.asm.q).apply(null,arguments)};
var La=c._free=function(){return(La=c._free=c.asm.r).apply(null,arguments)},Ka=c.___errno_location=function(){return(Ka=c.___errno_location=c.asm.s).apply(null,arguments)},la=c.stackSave=function(){return(la=c.stackSave=c.asm.t).apply(null,arguments)},I=c.stackAlloc=function(){return(I=c.stackAlloc=c.asm.u).apply(null,arguments)},ma=c.stackRestore=function(){return(ma=c.stackRestore=c.asm.v).apply(null,arguments)};c.dynCall_vi=function(){return(c.dynCall_vi=c.asm.w).apply(null,arguments)};
c.dynCall_v=function(){return(c.dynCall_v=c.asm.x).apply(null,arguments)};c.asm=Na;var Y;c.then=function(a){if(Y)a(c);else{var d=c.onRuntimeInitialized;c.onRuntimeInitialized=function(){d&&d();a(c)}}return c};P=function Oa(){Y||Z();Y||(P=Oa)};
function Z(){function a(){if(!Y&&(Y=!0,c.calledRun=!0,!ia)){M(ra);M(sa);if(c.onRuntimeInitialized)c.onRuntimeInitialized();if(c.postRun)for("function"==typeof c.postRun&&(c.postRun=[c.postRun]);c.postRun.length;){var d=c.postRun.shift();ta.unshift(d)}M(ta)}}if(!(0<N)){if(c.preRun)for("function"==typeof c.preRun&&(c.preRun=[c.preRun]);c.preRun.length;)ua();M(qa);0<N||(c.setStatus?(c.setStatus("Running..."),setTimeout(function(){setTimeout(function(){c.setStatus("")},1);a()},1)):a())}}c.run=Z;
if(c.preInit)for("function"==typeof c.preInit&&(c.preInit=[c.preInit]);0<c.preInit.length;)c.preInit.pop()();noExitRuntime=!0;Z();


  return Module
}
);
})();
if (typeof exports === 'object' && typeof module === 'object')
      module.exports = Module;
    else if (typeof define === 'function' && define['amd'])
      define([], function() { return Module; });
    else if (typeof exports === 'object')
      exports["Module"] = Module;
    export default (() => {
    const initialize = () => {
        return new Promise(resolve => {
            Module({
                locateFile() {
                    const i = import.meta.url.lastIndexOf('/')
                    return import.meta.url.substring(0, i) + '/glslang.wasm';
                },
                onRuntimeInitialized() {
                    resolve({
                        compileGLSLZeroCopy: this.compileGLSLZeroCopy,
                        compileGLSL: this.compileGLSL,
                    });
                },
            });
        });
    };

    let instance;
    return () => {
        if (!instance) {
            instance = initialize();
        }
        return instance;
    };
})();