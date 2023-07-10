package com.sabyasachi.finalimagedefencingapp

import android.content.Intent
import android.graphics.Bitmap
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.TextView
import android.widget.ImageView
import com.sabyasachi.finalimagedefencingapp.ml.MobilenetV110224Quant
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


        var imageProcessor= ImageProcessor.Builder().add(ResizeOp(224,224,ResizeOp.ResizeMethod.BILINEAR)).build()



        var labels=application.assets.open("labels.txt").bufferedReader().readLines()

        selectBtn.setOnClickListener{
            var intent = Intent()
            intent.setAction(Intent.ACTION_GET_CONTENT)
            intent.setType("image/*")
            startActivityForResult(intent,100)
        }

        predBtn.setOnClickListener{
            var tensorImage=TensorImage(DataType.UINT8)
            tensorImage.load(bitmap)

            tensorImage=imageProcessor.process(tensorImage)


            val model = MobilenetV110224Quant.newInstance(this)

            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
            inputFeature0.loadBuffer(tensorImage.buffer)


            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

            var maxIdx=0
            outputFeature0.forEachIndexed{index, fl ->
                if(outputFeature0[maxIdx]<fl)
                {
                    maxIdx=index
                }
            }

//            resView.setText(maxIdx.toString())
            resView.setText(labels[maxIdx])

            model.close()
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
}