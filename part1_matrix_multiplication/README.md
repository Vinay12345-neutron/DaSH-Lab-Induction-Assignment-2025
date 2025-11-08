
### 🧩 **Windows Compilation & Profiling Instructions (Nsight Systems + Nsight Compute). NOT WSL/Linux Compatible**

#### **1️⃣ Compile your CUDA program**

Open **PowerShell (as Administrator)** and run:

```bash
nvcc -O3 -arch=sm_86 -lcublas -o 2D_block_tiling.exe 2D_block_tiling.cu
```

> 📝 **Note:**
> If compiling on Windows, ensure this small code snippet is added at the top of your `.cu` file to avoid type errors:
>
> ```cpp
> #ifndef uint
> #define uint unsigned int
> #endif
> ```

---

#### **2️⃣ Profile using Nsight Systems**

In the same PowerShell window (Administrator mode), run:

```bash
nsys profile --trace=cuda,cublas,nvtx --sample=none -o 2D_block_tiling_report 2D_block_tiling.exe
```

This will generate a report file named:

```
2D_block_tiling_report.nsys-rep
```

---

#### **3️⃣ Analyze with Nsight Compute**

* Open **Nsight Compute (GUI)**.
* Go to **File → Open**, and select your generated `.nsys-rep` file.
* Navigate to the **Kernels** section which will be under CUDA HW.
* Zoom in when in Timeline View and **Right-click** on the desired kernel and select **"Profile Kernel"** and then click **Launch**. Then go to **Details**.
  ⚠️ **Do NOT start a new activity** or launch the executable again.


---

#### **4️⃣ Generate textual statistics (optional)**

To get a quick summary in text form:

```bash
nsys stats D:\Dash_assignment\DaSH-Lab-Induction-Assignment-2025\part1_matrix_multiplication\2D_block_tiling_report.nsys-rep
```

---

✅ **Summary**

* Compile → `nvcc -O3 -arch=sm_86 -lcublas -o exe file.cu`
* Profile → `nsys profile --trace=cuda,cublas,nvtx --sample=none -o report exe`
* Analyze → Open `.nsys-rep` in Nsight Compute → Right-click kernel → *Profile Kernel*
* Optional text stats → `nsys stats path\report.nsys-rep`

---