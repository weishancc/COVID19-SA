function callback_predict(response){
    var result = JSON.parse(response)["text"];
    $("#prediction").fadeOut("slow ",function(){
            $("#prediction").fadeIn("slow");
            document.getElementById("prediction").innerHTML = result;
    });
} 

function predict(){
    var sentiment = document.getElementById("sentence").value; 
    
    sentiment = {"text": sentiment};
    sentiment = JSON.stringify(sentiment);
    ajaxPostRequest("/predict", sentiment, callback_predict)
}


/*--------------------ajax--------------------*/
// path is URL we are sending request
// data is JSON blob being sent to the server
// callback function that JS calls when server replies

function ajaxGetRequest(path, callback){
    var request = new XMLHttpRequest();
    request.onreadystatechange = function(){
        if (this.readyState===4&&this.status ===200){
            callback(this.response);
        }
    };
    request.open("GET", path);
    request.send();
}

function ajaxPostRequest(path, data, callback){
    var request = new XMLHttpRequest();
    request.onreadystatechange = function(){
        if (this.readyState===4&&this.status ===200){
            callback(this.response);
        }
    };
    request.open("POST", path);
    request.send(data);
}
