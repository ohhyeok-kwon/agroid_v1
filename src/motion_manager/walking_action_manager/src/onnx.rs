//! Small Rust binding to the manufacturer's ONNX Runtime 1.18 C API (API 18).
//! Models and their tensor names are fixed, verified package assets of each action package.
use std::{
    ffi::{c_char, c_void, CStr, CString},
    path::PathBuf,
    ptr,
    sync::OnceLock,
};

type Handle = *mut c_void;
type Status = Handle;
#[repr(C)]
struct ApiBase {
    get_api: unsafe extern "C" fn(u32) -> *const *const c_void,
    version: unsafe extern "C" fn() -> *const c_char,
}

pub struct Session {
    api: *const *const c_void,
    env: Handle,
    options: Handle,
    session: Handle,
    memory: Handle,
}
// A Session is exclusively owned by the walking executor; Run is never concurrent.
unsafe impl Send for Session {}

impl Session {
    pub fn load(model: &[u8]) -> Result<Self, String> {
        let api = runtime_api()?;
        unsafe {
            let mut s = Self {
                api,
                env: ptr::null_mut(),
                options: ptr::null_mut(),
                session: ptr::null_mut(),
                memory: ptr::null_mut(),
            };
            let create_env: unsafe extern "C" fn(i32, *const c_char, *mut Handle) -> Status =
                s.function(3);
            let status = create_env(3, c"walking_action_manager".as_ptr(), &mut s.env);
            s.check(status)?;
            let create_options: unsafe extern "C" fn(*mut Handle) -> Status = s.function(10);
            let status = create_options(&mut s.options);
            s.check(status)?;
            let threads: unsafe extern "C" fn(Handle, i32) -> Status = s.function(24);
            s.check(threads(s.options, 1))?;
            let create: unsafe extern "C" fn(
                Handle,
                *const c_void,
                usize,
                Handle,
                *mut Handle,
            ) -> Status = s.function(8);
            let status = create(
                s.env,
                model.as_ptr().cast(),
                model.len(),
                s.options,
                &mut s.session,
            );
            s.check(status)?;
            let cpu: unsafe extern "C" fn(i32, i32, *mut Handle) -> Status = s.function(69);
            let status = cpu(1, 0, &mut s.memory);
            s.check(status)?;
            Ok(s)
        }
    }

    /// [1,810] observations -> [1,24] actions. Storage lives through synchronous Run.
    pub fn infer(&mut self, input: &mut [f32; 810]) -> Result<[f32; 24], String> {
        let mut output = [0.; 24];
        self.run(c"obs", input, c"actions", &mut output)?;
        Ok(output)
    }

    /// One float32 input [1,n] -> one float32 output holding exactly `output.len()` values.
    pub fn run(
        &mut self,
        input_name: &CStr,
        input: &mut [f32],
        output_name: &CStr,
        output: &mut [f32],
    ) -> Result<(), String> {
        if input.iter().any(|x| !x.is_finite()) {
            return Err("non-finite observation".into());
        }
        unsafe {
            let tensor: unsafe extern "C" fn(
                Handle,
                *mut c_void,
                usize,
                *const i64,
                usize,
                i32,
                *mut Handle,
            ) -> Status = self.function(49);
            let run: unsafe extern "C" fn(
                Handle,
                Handle,
                *const *const c_char,
                *const Handle,
                usize,
                *const *const c_char,
                usize,
                *mut Handle,
            ) -> Status = self.function(9);
            let shape = [1, input.len() as i64];
            let mut value = ptr::null_mut();
            self.check(tensor(
                self.memory,
                input.as_mut_ptr().cast(),
                input.len() * 4,
                shape.as_ptr(),
                2,
                1,
                &mut value,
            ))?;
            let mut result = ptr::null_mut();
            let status = self.check(run(
                self.session,
                ptr::null_mut(),
                [input_name.as_ptr()].as_ptr(),
                &value,
                1,
                [output_name.as_ptr()].as_ptr(),
                1,
                &mut result,
            ));
            self.release(96, value);
            if let Err(error) = status {
                self.release(96, result);
                return Err(error);
            }
            let status = self.read_output(result, output);
            self.release(96, result);
            status
        }
    }

    unsafe fn read_output(&self, value: Handle, output: &mut [f32]) -> Result<(), String> {
        let shape: unsafe extern "C" fn(Handle, *mut Handle) -> Status = self.function(65);
        let count: unsafe extern "C" fn(Handle, *mut usize) -> Status = self.function(64);
        let dtype: unsafe extern "C" fn(Handle, *mut i32) -> Status = self.function(60);
        let data: unsafe extern "C" fn(Handle, *mut Handle) -> Status = self.function(51);
        let mut info = ptr::null_mut();
        self.check(shape(value, &mut info))?;
        let mut n = 0;
        let mut kind = 0;
        let result = self
            .check(count(info, &mut n))
            .and_then(|_| self.check(dtype(info, &mut kind)));
        self.release(99, info);
        result?;
        if n != output.len() || kind != 1 {
            return Err(format!("expected {} float32 policy actions", output.len()));
        }
        let mut p = ptr::null_mut();
        self.check(data(value, &mut p))?;
        if p.is_null() {
            return Err("empty policy output".into());
        }
        output.copy_from_slice(std::slice::from_raw_parts(p.cast::<f32>(), output.len()));
        if output.iter().any(|v| !v.is_finite()) {
            return Err("non-finite policy action".into());
        }
        Ok(())
    }

    unsafe fn function<T: Copy>(&self, index: usize) -> T {
        // OrtApi is an append-only table of C function pointers. Indices are checked
        // against offsetof(OrtApi, ...) in the official v1.18.1 header.
        std::mem::transmute_copy(&*self.api.add(index))
    }
    unsafe fn release(&self, index: usize, handle: Handle) {
        if !handle.is_null() {
            let release: unsafe extern "C" fn(Handle) = self.function(index);
            release(handle);
        }
    }
    unsafe fn check(&self, status: Status) -> Result<(), String> {
        if status.is_null() {
            return Ok(());
        }
        let message: unsafe extern "C" fn(Handle) -> *const c_char = self.function(2);
        let text = CStr::from_ptr(message(status))
            .to_string_lossy()
            .into_owned();
        self.release(93, status);
        Err(text)
    }
}
impl Drop for Session {
    fn drop(&mut self) {
        unsafe {
            self.release(95, self.session);
            self.release(94, self.memory);
            self.release(100, self.options);
            self.release(92, self.env);
        }
    }
}

// ORT registers thread-local destructors: keep the library loaded for the process.
struct RuntimeApi {
    api: *const *const c_void,
    _library: Handle,
}
unsafe impl Send for RuntimeApi {}
unsafe impl Sync for RuntimeApi {}
fn runtime_api() -> Result<*const *const c_void, String> {
    static API: OnceLock<Result<RuntimeApi, String>> = OnceLock::new();
    API.get_or_init(|| {
        let path = library_path()?;
        let path = CString::new(path.to_string_lossy().as_bytes()).map_err(|e| e.to_string())?;
        unsafe {
            let library = libc::dlopen(path.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL);
            if library.is_null() {
                let error = libc::dlerror();
                return Err(if error.is_null() {
                    "cannot load ONNX Runtime".into()
                } else {
                    CStr::from_ptr(error).to_string_lossy().into_owned()
                });
            }
            let symbol = libc::dlsym(library, c"OrtGetApiBase".as_ptr());
            if symbol.is_null() {
                libc::dlclose(library);
                return Err("OrtGetApiBase is missing".into());
            }
            let get: unsafe extern "C" fn() -> *const ApiBase = std::mem::transmute(symbol);
            let api = ((*get()).get_api)(18);
            if api.is_null() {
                libc::dlclose(library);
                return Err("ONNX Runtime does not support API 18".into());
            }
            Ok(RuntimeApi {
                api,
                _library: library,
            })
        }
    })
    .as_ref()
    .map(|r| r.api)
    .map_err(Clone::clone)
}

fn library_path() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("AGROID_ONNX_RUNTIME") {
        return Ok(path.into());
    }
    let name = "libonnxruntime.so.1.18.1";
    if let Some(prefixes) = std::env::var_os("AMENT_PREFIX_PATH") {
        for prefix in std::env::split_paths(&prefixes) {
            let path = prefix
                .join("share/walking_action_manager/config")
                .join(name);
            if path.is_file() {
                return Ok(path);
            }
        }
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("config")
        .join(name);
    if path.is_file() {
        Ok(path)
    } else {
        Err("manufacturer ONNX Runtime is missing".into())
    }
}
