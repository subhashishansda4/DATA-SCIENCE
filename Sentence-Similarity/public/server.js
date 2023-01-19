const express = require("express");
const body_parser = require("body-parser");
const https = require("https");
const request = require("request")

const app = express();
app.use(express.static("static"));
app.use(body_parser.urlencoded({extended: true}));


app.get("/", function(req, res) {
    res.sendFile(__dirname + "/index.html");

})

app.post("/", function(req, res) {
    var data = {
        sentence1 : req.body.s1,
        sentence2 : req.body.s2
    }

    request.post({url:'http://localhost:5000/', form: data}, function(err, httpResponse, body) {
        var similarity = body

        if(similarity === 'similar') {
            res.sendFile(__dirname + "/similar.html");
        } else {
            res.sendFile(__dirname + "/unique.html");
        }
    })

})

app.post("/similar", function(req, res) {
    res.redirect("/");
})

app.post("/unique", function(req, res) {
    res.redirect("/");
})


app.listen(4000, () => {
    console.log('Server running on port 4000');
});