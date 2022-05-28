package com.example.facemaskdetection

import android.content.ContentValues.TAG
import android.content.pm.ModuleInfo
import android.content.pm.PackageManager
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.Camera
import android.graphics.ColorSpace
import android.os.Build
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.util.DisplayMetrics
import android.util.Log
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.appcompat.content.res.AppCompatResources
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.example.facemaskdetection.ml.FackMaskDetection
import com.google.common.util.concurrent.ListenableFuture
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.schema.Model
import java.lang.Double.max
import java.lang.Double.min
import java.lang.Exception
import java.lang.IllegalStateException
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.abs

typealias CameraBitMapOutputListener=(bitmap: Bitmap)->Unit

class MainActivity : AppCompatActivity() {
    private var preview : Preview?=null
    private var imageAnalyzer:  ImageAnalysis?=null
    private var lensFacing: Int = CameraSelector.LENS_FACING_FRONT
    private var camera: Camera? =null
    private lateinit var cameraExecutor:ExecutorService



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        setupML()
        setupCameraThread()
        setupCameraControllers()

        if(!allPermissionsGranted){
            requireCameraPermission()
        }
        else{
            setupCamera()
        }
    }

    private fun setupML() {
        val options: ColorSpace.Model.Options=
            Model.Options.Builder().setDevice(ColorSpace.Model.Device.GPU).setNumThreads(5)
            faceMaskDetection=FaceMaskDetection.newInstance(applicationContext,options)
    }

    private fun setupCamera() {
        val CameraProviderFuture:ListenableFuture<ProcessCameraProvider>=
        ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener(Runnable {
            cameraProvider=cameraProviderFuture.get()
            lensFacing=when{
                hasFrontCamera->CameraSelector.LENS_FACING_FRONT
                hasBackCamera->CameraSelector.LENS_FACING_BACK
                else->throw IllegalStateException("No Cameras Available")
            }
            setupCameraControllers()
            setupCameraUseCases()

        },ContextCompat.getMainExecutor(this)
    }
    private val allPermissionsGranted: Boolean
    get(){
      return  REQUIRED_PERMISSIONS.all{
            ContextCompat.checkSelfPermission(
                baseContext,it
            )==PackageManager.PERMISSION_GRANTED

        }
    }
    private val hasBackCamera: Boolean
    get(){
        return cameraProvider?.hasCamera(CameraSelector.DEFAULT_BACK_CAMERA)?:false
    }

    private val hasFrontCamera: Boolean
    get(){
        return cameraProvider?.hasCamera(CameraSelector.DEFAULT_FRONT_CAMERA)?:false
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        grantedCameraPermission(requestCode)
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        setupCameraControllers()
    }
    @RequiresApi(Build.VERSION_CODES.N)
    private fun aspectRatio(width:Int, height:Int):Int{
        val previewRatio:Double =max(width,height).toDouble()/min(width,height)
        if(abs(previewRatio-RATIO_4_3_VALUE)<=abs(previewRatio-RATIO_16_9_VALUE))
        {
            return AspectRatio.RATIO_4_3
        }
        return AspectRatio.RATIO_16_9
    }
    private lateinit var faceMaskDetection: FackMaskDetection

    private fun requireCameraPermission() {
        ActivityCompat.requestPermissions(
            this,REQUIRED_PERMISSIONS,REQUEST_CODE_PERMISSIONS
        )
    }

    private fun setupCameraThread(){
        cameraExecutor= Executors.newSingleThreadExecutor()
    }
    private fun setupCameraControllers(){
        fun setLensButtonIcon(){
            btn_camera_lens_face.setImageDrawable(
                AppCompatResources.getDrawable(
                    applicationContext,
                    if(lensFacing==CameraSelector.LENS_FACING_FRONT)
                        R.drawable.ic_baseline_camera_rear_24
                else
                    R.drawable.ic_baseline_camera_front_24
                )
            )
        }
        setLensButtonIcon()
        btn_camera_lens_face.setOnClickListener{
            lensFacing=if (CameraSelector.LENS_FACING_FRONT==lensFacing){
                CameraSelector.LENS_FACING_BACK
            }
            else{
                CameraSelector.LENS_FACING_FRONT
            }
            setLensButtonIcon()
            setupCameraUseCases()
        }
        try{
            btn_camera_lens_face.isEnabled=hasBackCamera && hasFrontCamera
        }
        catch (exception:CameraInfoUnavailableException){
            btn_camera_lens_face.isEnabled=false
        }

    }

    private fun setupCameraUseCases() {
        val cameraSelector:CameraSelector=
            CameraSelector.Builder.requiredLensFacing(lensFacing).build()
        val metrics: DisplayMetrics=
            DisplayMetrics().also{preview_view.display.getRealMetrics(it)}
        val rotation:Int =preview_view.display.rotation
        val screenAspectRatio: Int =aspectRatio(metrics.widthPixels,metrics.heightPixels)
        preview= Preview.Builder()
            .setTargetAspectRatio(screenAspectRatio)
            .setTargetRotation(rotation)
            .build()

        imageAnalyzer=ImageAnalysis.Builder()
            .setTargetAspectRatio(screenAspectRatio)
            .setTargetRotation(rotation)
            .build()
            .also {
                it.setAnalyzer(
                    cameraExecutor, BitmapOutPutAnalysis(aplicationContext) { bitmap ->
                        setupMLOutput(bitmap)
                    }
                )
            }
      cameraProvider?.unbindAll()
        try {
            camera = cameraProvider?.bindToLifecycle(
                this, cameraSelector, preview, imageAnalyzer
            )
            preview?.setSurfaceProvider(preview_view.createSurfaceProvider())
        }
        catch (exc:Exception){
            Log.e(TAG,"USE CASE BINDING FAILURE",exc)

        }
    }

    private fun setupMLOutput(bitmap: Bitmap) {
        val tensorImage: TensorImage=TensorImage.fromBitmap(bitmap)
        val result: FaceMaskDetection Outputs = faceMaskDetection.process(tensorImage)

        val output:List<Locale.Category>=
        result.probabilityAsCategoryList.apply{
            sortByDescending{res ->res.score}
            lifecycleScope.launching(Dispatchers.Main){
                category->
                tv_output.text=category.label
                tv_output.setTextColor(
                    ContextCompat.getColor(
                        applicationContext,
                        if(category.label=="without_mask")R.color.red
                    else
                        R.color.green
                    )
                )
                overlay.background=getDrawable(
                    if(category.label=="without_mask")R.drawable.red_border
                else
                    R.drawable.green_border
                )
                pb_output.progressTintList=AppCompatResources.getColorStateList(
                    applicationContext,
                    if(category.label=="without_mask")R.color.red
                else
                    R.color.green
                )
                pb_output.progress=(category.score*100).toInt()

            }
        }

    }


    private fun setupCameraThread(){
        cameraExecutor=Executors.newSingleThreadExecutor()
    }
    private fun grantedCameraPermission(requestCode: Int){
        if(requestCode==REQUEST_CODE_PERMISSIONS){
            if(!allPermissionsGranted){
                setupCamera()
            }
            else{
                Toast.makeText(this,"Permission NOT granted",Toast.LENGTH_SHORT).show()
                finish()
            }

        }
    }

}