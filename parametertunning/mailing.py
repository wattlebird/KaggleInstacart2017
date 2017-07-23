from pprint import PrettyPrinter
import smtplib
from email.message import EmailMessage
from datetime import datetime
import pandas as pd


def mailsend(experiment, parameter, targetname, target, f1, auc):
    msg = EmailMessage()
    # build table
    paramtable = pd.DataFrame({
        'F1': f1,
        'AUC': auc
    }, columns=['F1', 'AUC'], index=pd.Index(target, name=targetname))
    msg.set_content("""
    <html>
      <head></head>
      <body>
        {0}
        Bonjour! The experiment "{1}" parameter tunning has completed. 

        Parameter setting:
        {2}

        Cross-validation Tunning Result:
        {3}

        Please find the debug files on Azure Blob Service.abs

        Yours sincerely
        Team parameter tunning server
      </body>
    </html>
    """.format(datetime.today().strftime("%a, %d %b %Y %H:%M:%S"), experiment, parameter, paramtable.to_html()))

    msg['Subject'] = "Parameter Tunning Completed."
    msg['From'] = "azureuser@cloudymiao.cloudapp.net"
    msg['To'] = "geniusxiaoguai@gmail.com"

    # Send the message via our own SMTP server.
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()


def mailsendfail(experiment, parameter, exception):

    msg = EmailMessage()
    msg.set_content("""
    {0}
    Bonjour! The experiment "{1}" parameter tunning has failed. 

    Parameter:
    {2}

    Exception message:
    {3}

    Yours sincerely
    Team parameter tunning server

    """.format(datetime.today().strftime("%a, %d %b %Y %H:%M:%S"), experiment, parameter, str(exception)))

    msg['Subject'] = "Parameter Tunning Failed."
    msg['From'] = "azureuser@cloudymiao.cloudapp.net"
    msg['To'] = "geniusxiaoguai@gmail.com"

    # Send the message via our own SMTP server.
    s = smtplib.SMTP('localhost')
    s.send_message(msg)
    s.quit()