SampleText_VI = "Một con mèo nhỏ bị ném từ một chiếc xe hơi trên đường vành đai Toulouse" + "\n" + "\n" + "Hiệp hội bảo vệ động vật tìm thấy con mèo nhỏ bị thương. Điều này đang gióng lên hồi chuông cảnh báo về sự gia tăng việc bỏ rơi thú cưng trong mùa hè." + "\n" + "\n" + "Một cử chỉ khó hiểu và tàn nhẫn. Một con mèo ba tháng tuổi, được gọi là Nounette, đã bị cố tình ném từ một chiếc xe đang di chuyển vào cuối tháng 7 trên đường vành đai Toulouse, báo cáo của France Bleu. Đó là một hiệp hội bảo vệ động vật địa phương, Cha'Mania tiết lộ câu chuyện để cảnh báo số lượng động vật bị bỏ rơi ngày càng tăng trong mùa hè." + "\n" + "\n" + "Sự thật xảy ra trên đường vành đai Toulouse vào ngày 27 tháng 7. Sau khi đuổi con vật ra khi lái xe, tài xế quay xe lại và đi mất. Nounette (mèo) gần như đã chết. Cô bị một phương tiện khác đâm trúng và phát hiện bị thương nặng trong bụi rậm bởi một người. Người này sau đó đã đưa cô đến cơ sở của Cha'Mania để điều trị." + "\n" + "\n" + "«Mang đến một bác sĩ thú y, chúng tôi nhận thấy rằng bên trong cơ thể cô ấy, mọi thứ đều bị phá vỡ: chú mèo con có ba vết nứt ở xương chậu. Để có được trở lại trên đôi chân của mình, cô được kê đơn thuốc kháng sinh và được điều trị bằng lồng, » đài phát thanh nói." + "\n" + "\n" + "«Con vật không phải là một món đồ nội thất»" + "\n" + "\n" + "Tức giận, các thành viên của hiệp hội muốn sử dụng sự kiện này để nâng cao nhận thức của công chúng về việc bỏ rơi và lạm dụng vật nuôi, một hiện tượng mà họ cảm thấy là quá phổ biến. «Thật đơn giản để gọi cho hiệp hội và nói rằng, bạn không muốn thú cưng của bạn vì một số lý do gì đó, nhưng từ đó ném nó ra ngoài cửa sổ trên đường, lái xe ... Rõ ràng là thiếu sự tôn trọng đối với động vật », Brigitte Maréchaux, một tình nguyện viên ở France Bleu đã phản ứng. Chính người phụ nữ này đã trông coi con mèo vào cho đến khi nó bình phục." + "\n" + "\n" + "«Chúng ta sẽ phải đưa ra một luật. Mỗi năm, chúng tôi tiếp nhận ngày càng nhiều động vật bị bỏ rơi, một tình nguyện viên nói với La Dépêche du Midi. Nó phải dừng lại. Một con vật không phải là một món đồ nội thất ». Hiệp hội đã đưa ra một lời kêu gọi các nhân chứng với hy vọng xác định người chịu trách nhiệm và nộp đơn kiện chống lại anh ta." + "\n" + "\n" + "Nguồn: https://goo.gl/Z1ACXA";
langSite = 'vi';
SampleText = SampleText_VI;
urlAPI = 'https://resoomer.com/';
limit_char_transfert = 10000;
max_char_post = 200000;
issetPost = 0;

setExemple = function () {
    var textExemple = SampleText;
    $("#contentText").val(textExemple);
    if (window.screen.availWidth >= 500) {
        document.getElementById("contentText").style.height = "1px";
        document.getElementById("contentText").style.height = (25 + document.getElementById("contentText").scrollHeight) + "px";
    }
    noticeForText();
};

clearTextarea = function () {
    $("#contentText").val("");
    $("#contentText").keyup();
    $("#returnResoomer").html("");
    $(divADS).addClass("DisplayNone");
    $("div.adsFeature").addClass("DisplayNone");
    noticeTextSize();
    // $.ajax({
    //     encoding: "UTF-8",
    //     url: urlAPI + 'controllers/AFC.php?action=clearText',
    //     success: function () { }
    // });
};

noticeForText = function () {
    var contentText = $("#contentText").val();
    if (contentText == "") {
        $("#noticeForText").animate({ fontSize: '12px', fontWeight: 200 }, 125);
    } else {
        $("#noticeForText").animate({ fontSize: '13px', fontWeight: 600 }, 125);
    }
};

noticeTextSize = function () {
    /*
    var textValue = $('#contentText').val();
    var textSize = textValue.length;
    if(textSize > max_char_post)	{
        $('#noticeSizeText').html(textSize);
        $('#conteneurNoticeSizeText').removeClass('DisplayNone');
    }else{
        $('#conteneurNoticeSizeText').addClass('DisplayNone');
    }
    */
};

function textAreaAdjust(o) {
    if (window.screen.availWidth >= 500) {
        o.style.height = "1px";
        o.style.height = (25 + o.scrollHeight) + "px";
    }
    var contentText = $("#contentText").val();
    noticeForText();
    noticeTextSize();
}
