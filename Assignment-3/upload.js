function uploadAndClassifyImageWithResnet34() {
var fileInput = document.getElementById('resnet34FileUpload').files;
if(!fileInput.length) {
	return alert("Please choose a file to upload first.");
}
var file=fileInput[0];
var filename=file.name;
var formData= new FormData();
formData.append(filename,file);

console.log(filename);
$.ajax({
	async: true,
	crossDomain: true,
	method: 'POST',
	url: 'https://njvi27x0p9.execute-api.ap-south-1.amazonaws.com/dev/classify',
	data: formData,
	processData: false,
	contentType: false,
	mimeType: "multipart/form-data",
})
.done(function (response) {
	console.log(response);
	document.getElementById('resnet34Result').textContent = response;
})
.fail(function(){
	alert("There was an error while sending prediction request to resnet34 model.");
});
};

function uploadAndClassifyImageWithMobilenetv2() {
var fileInput = document.getElementById('mobilenetv2FileUpload').files;
if(!fileInput.length) {
	return alert("Please choose a file to upload first.");
}
var file=fileInput[0];
var filename=file.name;
var formData= new FormData();
formData.append(filename,file);

console.log(filename);
$.ajax({
	async: true,
	crossDomain: true,
	method: 'POST',
	url: 'https://s3b87cmdqe.execute-api.ap-south-1.amazonaws.com/dev/classify',
	data: formData,
	processData: false,
	contentType: false,
	mimeType: "multipart/form-data",
})
.done(function (response) {
	console.log(response);
	document.getElementById('mobilenetv2Result').textContent = response;
})
.fail(function(){
	alert("There was an error while sending prediction request to MobileNetV2 model.");
});
};

function uploadAndAlignImage() {
var fileInput = document.getElementById('dlibImageAlign').files;
if(!fileInput.length) {
	return alert("Please choose a file to upload first.");
}
var file=fileInput[0];
var filename=file.name;
var formData= new FormData();
formData.append(filename,file);

console.log(filename);
$.ajax({
	async: true,
	crossDomain: true,
	method: 'POST',
	url: 'https://0j8yn8vbqh.execute-api.ap-south-1.amazonaws.com/dev/align',
	data: formData,
	processData: false,
	contentType: false,
	mimeType: "multipart/form-data",
})
.done(function (response) {
	console.log(response);
	pred=JSON.parse(response).predicted;
	var height=pred.length;
	var width=pred[0].length;
	var buffer = new Uint8ClampedArray(width * height * 4);
	for(var y = 0; y < height; y++) {
    for(var x = 0; x < width; x++) {
        var pos = (y * width + x) * 4; // position in buffer based on x and y
        buffer[pos  ] = pred[y][x][2];           // some R value [0, 255]
        buffer[pos+1] = pred[y][x][1];           // some G value
        buffer[pos+2] = pred[y][x][0];           // some B value
        buffer[pos+3] = 240;           // set alpha channel
		}
	}
	
	var canvas = document.createElement('canvas'),
    ctx = canvas.getContext('2d');
	canvas.width = width;
	canvas.height = height;
	var idata = ctx.createImageData(width, height);
	idata.data.set(buffer);
	ctx.putImageData(idata, 0, 0);
	//ctx.drawImage(image,0,0,width,height); 
	var dataUri = canvas.toDataURL();
	var image = new Image;
	image.src = dataUri;
	
	document.getElementById('alignImageResult').innerHTML = '';
	document.getElementById('alignImageResult').appendChild(image);
	//document.getElementById('alignImageResult').style.backgroundImage=url(dataUri);
})
.fail(function(error){
	console.log(error);
	alert("There was an error while sending prediction request to Align Image model.");
});
};

function uploadAndSwapFace() {
var fileInput1 = document.getElementById('dlibSwapFace1').files;
if(!fileInput1.length) {
	return alert("Please choose a file to upload first.");
}
var file1=fileInput1[0];
var filename1=file1.name;
var formData= new FormData();
formData.append(filename1,file1);

var fileInput2 = document.getElementById('dlibSwapFace2').files;
if(!fileInput2.length) {
	return alert("Please choose a file to upload first.");
}
var file2=fileInput2[0];
var filename2=file2.name;
//var formData= new FormData();
formData.append(filename2,file2);

console.log(filename1);
console.log(filename2);
$.ajax({
	async: true,
	crossDomain: true,
	method: 'POST',
	url: 'https://1ptmf9i3rl.execute-api.ap-south-1.amazonaws.com/dev/swap_face',
	data: formData,
	processData: false,
	contentType: false,
	mimeType: "multipart/form-data",
})
.done(function (response) {
	console.log(response);
	pred=JSON.parse(response).predicted;
	var height=pred.length;
	var width=pred[0].length;
	var buffer = new Uint8ClampedArray(width * height * 4);
	for(var y = 0; y < height; y++) {
    for(var x = 0; x < width; x++) {
        var pos = (y * width + x) * 4; // position in buffer based on x and y
        buffer[pos  ] = pred[y][x][2];           // some R value [0, 255]
        buffer[pos+1] = pred[y][x][1];           // some G value
        buffer[pos+2] = pred[y][x][0];           // some B value
        buffer[pos+3] = 240;           // set alpha channel
		}
	}
	
	var canvas = document.createElement('canvas'),
    ctx = canvas.getContext('2d');
	canvas.width = width;
	canvas.height = height;
	var idata = ctx.createImageData(width, height);
	idata.data.set(buffer);
	ctx.putImageData(idata, 0, 0);
	//ctx.drawImage(image,0,0,width,height); 
	var dataUri = canvas.toDataURL();
	var image = new Image;
	image.src = dataUri;
	 
	document.getElementById('swapFaceResult').innerHTML = '';
	document.getElementById('swapFaceResult').appendChild(image);
	//document.getElementById('alignImageResult').style.backgroundImage=url(dataUri);
})
.fail(function(error){
	console.log(error);
	alert("There was an error while sending prediction request to Align Image model.");
});
};

function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();

        reader.onload = function (e) {
            $('#blah').attr('src', e.target.result);
        }

        reader.readAsDataURL(input.files[0]);
    }
}

$('#btnResNetUpload').click(uploadAndClassifyImageWithResnet34);
$('#btnMobilenetv2Upload').click(uploadAndClassifyImageWithMobilenetv2);
$('#btAlignImageUpload').click(uploadAndAlignImage);
$('#btSwapFaceUpload').click(uploadAndSwapFace);
$("#dlibImageAlign").change(function(){
    readURL(this);
});