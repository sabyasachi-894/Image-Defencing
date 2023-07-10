package com.sabyasachi.imagetoimageapp

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.sabyasachi.imagetoimageapp.ml.TempModel
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    lateinit var selectBtn: Button
    lateinit var predBtn: Button
    lateinit var resView: TextView
    lateinit var imageView: ImageView
    lateinit var bitmap: Bitmap




    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)



        selectBtn = findViewById(R.id.selectBtn)
        predBtn = findViewById(R.id.predictBtn)
        resView=findViewById(R.id.resView)
        imageView=findViewById(R.id.imageView)

        lateinit var model: TempModel


        selectBtn.setOnClickListener{
            var intent = Intent()
            intent.setAction(Intent.ACTION_GET_CONTENT)
            intent.setType("image/*")
            startActivityForResult(intent,100)
        }


        model = TempModel.newInstance(this)

        predBtn.setOnClickListener{

            // Create the input tensor from the bitmap
            val inputImageBuffer = TensorImage(DataType.FLOAT32)
            inputImageBuffer.load(bitmap)

            // Preprocess the input image
            val imageProcessor = ImageProcessor.Builder()
                .add(ResizeOp(256, 256, ResizeOp.ResizeMethod.BILINEAR))
                // Add any other required preprocessing operations here
                .build()

            val preprocessedImage = imageProcessor.process(inputImageBuffer)


            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 256, 256, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(preprocessedImage.buffer)
            print("Message  ")
            print(inputFeature0)

            // Runs model inference and gets the result
            val outputs = model.process(inputFeature0)
            val outputImageBuffer = outputs.outputFeature0AsTensorBuffer

            // Assuming you have a TensorBuffer named 'outputImageBuffer'
            val outputValue = outputImageBuffer.floatArray

            Log.d("Output", "Output Value: ${outputValue.contentToString()}")

            // Convert the output tensor to a bitmap
            //val outputBitmap = outputImageBuffer.bitmap

            // Display the output bitmap in the imageView
            //imageView.setImageBitmap(outputBitmap)

            // Convert the output tensor to a bitmap
            val outputBitmap = tensorBufferToBitmap(outputImageBuffer)

            // Display the output bitmap in the imageView
            imageView.setImageBitmap(outputBitmap)

        }



    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if(requestCode==100)
        {
            var uri=data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver,uri)
            imageView.setImageBitmap(bitmap)
        }
    }




    private fun tensorBufferToBitmap(tensorBuffer: TensorBuffer): Bitmap {
        val shape = tensorBuffer.shape
        val width = shape[1]
        val height = shape[2]
        val intValues = IntArray(width * height)
        val floatBuffer = tensorBuffer.buffer.asFloatBuffer()

        for (i in 0 until width) {
            for (j in 0 until height) {
                val r = (floatBuffer.get((i * height + j) * 3) * 255).toInt()
                val g = (floatBuffer.get((i * height + j) * 3 + 1) * 255).toInt()
                val b = (floatBuffer.get((i * height + j) * 3 + 2) * 255).toInt()
                intValues[i * height + j] = Color.rgb(r, g, b)
            }
        }

        return Bitmap.createBitmap(intValues, width, height, Bitmap.Config.ARGB_8888)
    }

}