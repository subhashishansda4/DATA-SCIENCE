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
        game_name : req.body.gn,
        tag_line : req.body.tl
    }

    request.post({url:'http://localhost:5000/', form: data}, function(err, httpResponse, body) {
        var puuid = body

        if(puuid === '400') {
            res.sendFile(__dirname + "/error.html");
        } else {
            res.sendFile(__dirname + "/result.html");
        }
    })
})

app.post("/error", function(req, res) {
    res.redirect("/");
})
app.post("/result", function(req, res) {
    res.redirect("/");
})


app.listen(4000, () => {
    console.log('Server running on port 4000');
});